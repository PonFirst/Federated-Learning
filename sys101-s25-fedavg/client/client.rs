use candle_core::{DType, Result as CandleResult, Tensor, D, Module, Device};
use candle_nn::{loss, ops, Optimizer, VarBuilder, VarMap, SGD};
use candle_datasets::vision::Dataset;
use candle_app::{LinearModel, Model};
use tokio::net::{TcpStream, TcpListener};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use base64::Engine;
use anyhow::Result;
use std::sync::Arc;
use rand::seq::SliceRandom;
use rand::thread_rng;
use tokio::signal;

struct Client {
    server_addr: String,
    model: Option<(LinearModel, VarMap)>,
    model_name: String,
    status: String,
    dataset: Option<Arc<Dataset>>,
    local_addr: String,
}

impl Client {
    fn new(server_addr: &str, model_name: &str) -> Self {
        Client {
            server_addr: server_addr.to_string(),
            model: None,
            model_name: model_name.to_string(),
            status: "initialized".to_string(),
            dataset: None,
            local_addr: String::new(),
        }
    }

    async fn join(&mut self, server_ip: &str, model: &str) -> Result<TcpStream> {
        let mut stream = TcpStream::connect(server_ip).await?;

        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let local_addr = listener.local_addr()?.to_string();
        self.local_addr = local_addr.clone();

        let message = format!("REGISTER|{}|{}", local_addr, model);
        stream.write_all(message.as_bytes()).await?;
        stream.flush().await?;

        let mut buffer = [0; 1024];
        let n = stream.read(&mut buffer).await?;
        let response = String::from_utf8_lossy(&buffer[..n]);
        println!("Server response: {}", response);
        self.model_name = model.to_string();

        let mut dataset = candle_datasets::vision::mnist::load()?;
        let train_size = 10000;
        let num_samples = dataset.train_images.dims()[0];
        let mut indices: Vec<usize> = (0..num_samples).collect();
        indices.shuffle(&mut thread_rng());
        let train_indices = Tensor::from_vec(
            indices[..train_size].to_vec().into_iter().map(|x| x as u32).collect(),
            train_size,
            &Device::Cpu,
        )?;
        dataset.train_images = dataset.train_images.index_select(&train_indices, 0)?;
        dataset.train_labels = dataset.train_labels.index_select(&train_indices, 0)?;
        let dataset_arc = Arc::new(dataset);
        self.dataset = Some(dataset_arc.clone());

        stream.write_all(b"READY").await?;
        stream.flush().await?;

        let dataset_clone = dataset_arc.clone();
        tokio::spawn(async move {
            if let Err(e) = Self::run_inner(listener, dataset_clone).await {
                eprintln!("Client run error: {}", e);
            }
        });

        Ok(stream)
    }

    async fn run_inner(listener: TcpListener, dataset: Arc<Dataset>) -> Result<()> {
        println!("Client listening on {}", listener.local_addr()?);

        loop {
            let (mut client_stream, _) = listener.accept().await?;
            let mut buffer = [0; 65536];
            match client_stream.read(&mut buffer).await {
                Ok(n) => {
                    let message = String::from_utf8_lossy(&buffer[..n]);
                    if message == "COMPLETE" {
                        println!("Received from server: Training completed");
                    } else {
                        let parts: Vec<&str> = message.split('|').collect();
                        if parts[0] == "TRAIN" && parts.len() == 4 {
                            println!("Received TRAIN request for {}", parts[1]);
                            let weights_data: Vec<f32> = bincode::deserialize(
                                &base64::engine::general_purpose::STANDARD.decode(parts[2])?,
                            )?;
                            let bias_data: Vec<f32> = bincode::deserialize(
                                &base64::engine::general_purpose::STANDARD.decode(parts[3])?,
                            )?;

                            let weights = Tensor::from_vec(weights_data, &[10, 784], &Device::Cpu)?;
                            let bias = Tensor::from_vec(bias_data, &[10], &Device::Cpu)?;
                            let varmap = VarMap::new();
                            let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
                            let mut model = LinearModel::new(vs)?;
                            {
                                let mut data = varmap.data().lock().unwrap();
                                data.get_mut("linear.weight").unwrap().set(&weights)?;
                                data.get_mut("linear.bias").unwrap().set(&bias)?;
                            }

                            train_model(&mut model, &varmap, &dataset, 10).await?;

                            let weights_data = model.weight()?.to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
                            let bias_data = model.bias()?.to_vec1::<f32>()?;

                            let response = format!(
                                "UPDATE|{}|{}",
                                base64::engine::general_purpose::STANDARD.encode(&bincode::serialize(&weights_data)?),
                                base64::engine::general_purpose::STANDARD.encode(&bincode::serialize(&bias_data)?)
                            );
                            client_stream.write_all(response.as_bytes()).await?;
                            client_stream.flush().await?;
                        } else {
                            println!("Received message: {}", message);
                        }
                    }
                }
                Err(e) => eprintln!("Error reading from server: {}", e),
            }
        }
    }

    async fn train(&mut self, dataset: &Dataset, epochs: usize) -> CandleResult<()> {
        self.status = "training".to_string();
        let dev = Device::Cpu;
        let train_images = dataset.train_images.to_device(&dev)?;
        let train_labels = dataset.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
        let test_images = dataset.test_images.to_device(&dev)?;
        let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

        let (model, varmap) = self.model.as_mut().ok_or_else(|| candle_core::Error::Msg("No model available".into()))?;
        let mut sgd = SGD::new(varmap.all_vars(), 0.1)?;

        for epoch in 1..=epochs {
            let logits = Model::forward(model, &train_images)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_labels)?;
            sgd.backward_step(&loss)?;

            let test_logits = Model::forward(model, &test_images)?;
            let sum_ok = test_logits
                .argmax(D::Minus1)?
                .eq(&test_labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            let test_accuracy = sum_ok / test_labels.dims1()? as f32;
            println!(
                "Epoch {}: loss {:8.5}, test acc {:5.2}%",
                epoch,
                loss.to_scalar::<f32>()?,
                100. * test_accuracy
            );
        }
        self.status = "ready".to_string();
        Ok(())
    }

    fn get(&self) -> Option<(&LinearModel, &str)> {
        self.model.as_ref().map(|(m, _)| (m, self.status.as_str()))
    }

    fn test(&self, dataset: &Dataset) -> CandleResult<f32> {
        let dev = Device::Cpu;
        let test_images = dataset.test_images.to_device(&dev)?;
        let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

        let (model, _) = self.model.as_ref().ok_or_else(|| candle_core::Error::Msg("No model available".into()))?;
        let logits = Model::forward(model, &test_images)?;
        let sum_ok = logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let accuracy = sum_ok / test_labels.dims1()? as f32;
        Ok(accuracy)
    }
}

async fn train_model(model: &mut LinearModel, varmap: &VarMap, dataset: &Dataset, epochs: usize) -> CandleResult<()> {
    let dev = Device::Cpu;
    let train_images = dataset.train_images.to_device(&dev)?;
    let train_labels = dataset.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let test_images = dataset.test_images.to_device(&dev)?;
    let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let mut sgd = SGD::new(varmap.all_vars(), 0.1)?;

    for epoch in 1..=epochs {
        let logits = Model::forward(model, &train_images)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;
        sgd.backward_step(&loss)?;

        let test_logits = Model::forward(model, &test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "Epoch {}: loss {:8.5}, test acc {:5.2}%",
            epoch,
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut client = Client::new("127.0.0.1:50051", "mnist");
    let stream = client.join("127.0.0.1:50051", "mnist").await?;
    println!("Client setup complete on {}", client.local_addr);
    println!("Client running. Press Ctrl+C to terminate.");

    tokio::select! {
        _ = async {
            loop {
                let mut buffer = [0; 1024];
                match stream.try_read(&mut buffer) {
                    Ok(n) if n > 0 => {
                        let response = String::from_utf8_lossy(&buffer[..n]);
                        if response == "Training completed" {
                            println!("Received from server: Training completed");
                        } else {
                            println!("Received from server: {}", response);
                        }
                    }
                    Ok(_) => tokio::time::sleep(tokio::time::Duration::from_millis(100)).await,
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                    Err(e) => {
                        eprintln!("Error reading from server: {}", e);
                        break;
                    }
                }
            }
            Ok::<(), anyhow::Error>(())
        } => {}
        _ = signal::ctrl_c() => {
            println!("Received Ctrl+C, shutting down client.");
        }
    }

    println!("Client terminated.");
    Ok(())
}