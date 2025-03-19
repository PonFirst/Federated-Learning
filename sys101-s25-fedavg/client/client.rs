use candle_core::{DType, Result as CandleResult, Tensor, D, IndexOp, Module};
use candle_nn::{loss, ops, Optimizer, VarBuilder, VarMap, SGD};
use candle_datasets::vision::Dataset;
use candle_app::LinearModel;
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use base64::Engine;
use anyhow::Result;
use std::sync::Arc;

struct Client {
    server_addr: String,
    model: Option<(LinearModel, VarMap)>,
    model_name: String,
    status: String,
    dataset: Option<Arc<Dataset>>,
}

impl Client {
    fn new(server_addr: &str, model_name: &str) -> Self {
        Client {
            server_addr: server_addr.to_string(),
            model: None,
            model_name: model_name.to_string(),
            status: "initialized".to_string(),
            dataset: None,
        }
    }

    async fn join(&mut self, server_ip: &str, model: &str) -> Result<TcpStream> {
        let mut stream = TcpStream::connect(server_ip).await?;
        let message = format!("REGISTER|{}|{}", "127.0.0.1:50001", model);
        stream.write_all(message.as_bytes()).await?;
        stream.flush().await?;
    
        let mut buffer = [0; 1024];
        let n = stream.read(&mut buffer).await?;
        let response = String::from_utf8_lossy(&buffer[..n]);
        println!("Server response: {}", response);
        self.model_name = model.to_string();
        self.dataset = Some(Arc::new(candle_datasets::vision::mnist::load()?));
        Ok(stream)
    }

    async fn run(&mut self, mut stream: TcpStream) -> Result<()> {
        let request = format!("GET|{}", self.model_name);
        stream.write_all(request.as_bytes()).await?;
        stream.flush().await?;
    
        let mut buffer = [0; 65536];
        loop {
            match stream.read(&mut buffer).await {
                Ok(0) => {
                    println!("Server disconnected");
                    break;
                }
                Ok(n) => {
                    let response = String::from_utf8_lossy(&buffer[..n]);
                    println!("Received from server: {}", response);
                    let parts: Vec<&str> = response.split('|').collect();
                    if parts.len() >= 3 && parts[0] == "MODEL" {
                        let weights_bytes = base64::engine::general_purpose::STANDARD.decode(parts[1])?;
                        let bias_bytes = base64::engine::general_purpose::STANDARD.decode(parts[2])?;
                        let weights_data: Vec<f32> = bincode::deserialize(&weights_bytes)?;
                        let bias_data: Vec<f32> = bincode::deserialize(&bias_bytes)?;
    
                        let weights = Tensor::from_vec(weights_data, &[10, 784], &candle_core::Device::Cpu)?;
                        let bias = Tensor::from_vec(bias_data, &[10], &candle_core::Device::Cpu)?;
    
                        let varmap = VarMap::new();
                        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &candle_core::Device::Cpu);
                        let model = LinearModel::new(vs)?;
    
                        {
                            let mut data = varmap.data().lock().unwrap();
                            data.get_mut("linear.weight")
                                .expect("linear.weight missing")
                                .set(&weights)?;
                            data.get_mut("linear.bias")
                                .expect("linear.bias missing")
                                .set(&bias)?;
                        }
    
                        self.model = Some((model, varmap));
                        self.status = "ready".to_string();
                        println!("Global model received and loaded");
    
                        let dataset_arc = self.dataset.clone().unwrap();
                        self.train(&*dataset_arc, 15, true).await?;
    
                        let (_, varmap) = self.model.as_ref().unwrap();
                        let data = varmap.data().lock().unwrap();
                        let weights_tensor = data.get("linear.weight").unwrap();
                        let bias_tensor = data.get("linear.bias").unwrap();
                        let weights_data = weights_tensor.to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
                        let bias_data = bias_tensor.to_vec1::<f32>()?;
                        let response = format!(
                            "UPDATE|{}|{}",
                            base64::engine::general_purpose::STANDARD.encode(&bincode::serialize(&weights_data)?),
                            base64::engine::general_purpose::STANDARD.encode(&bincode::serialize(&bias_data)?)
                        );
                        stream.write_all(response.as_bytes()).await?;
                        stream.flush().await?;
                    } else if response.trim() == "Update received" {
                        println!("Server acknowledged update");
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Error reading from server: {}", e);
                    break;
                }
            }
        }
        Ok(())
    }
    

    async fn train(&mut self, dataset: &Dataset, epochs: usize, half: bool) -> CandleResult<()> {
        self.status = "training".to_string();
        let dev = candle_core::Device::Cpu;
        let train_images = dataset.train_images.to_device(&dev)?;
        let train_labels = dataset.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
        let test_images = dataset.test_images.to_device(&dev)?;
        let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

        let (train_images, train_labels, test_images, test_labels) = if half {
            (
                train_images.i(..train_images.dims()[0] / 2)?,
                train_labels.i(..train_labels.dims()[0] / 2)?,
                test_images.i(..test_images.dims()[0] / 2)?,
                test_labels.i(..test_labels.dims()[0] / 2)?,
            )
        } else {
            (train_images, train_labels, test_images, test_labels)
        };

        let (model, varmap) = self.model.as_mut().ok_or_else(|| candle_core::Error::Msg("No model available".into()))?;
        let mut sgd = SGD::new(varmap.all_vars(), 0.01)?;

        for epoch in 1..=epochs {
            let logits = model.forward(&train_images)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_labels)?;
            sgd.backward_step(&loss)?;

            let test_logits = model.forward(&test_images)?;
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
        let dev = candle_core::Device::Cpu;
        let test_images = dataset.test_images.to_device(&dev)?;
        let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

        let (model, _) = self.model.as_ref().ok_or_else(|| candle_core::Error::Msg("No model available".into()))?;
        let logits = model.forward(&test_images)?;
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

#[tokio::main]
async fn main() -> Result<()> {
    let mut client = Client::new("127.0.0.1:50051", "mnist");
    let stream = client.join("127.0.0.1:50051", "mnist").await?;
    println!("Client setup complete.");
    client.run(stream).await?;
    println!("Client terminated.");
    Ok(())
}