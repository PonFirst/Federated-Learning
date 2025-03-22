use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use candle_core::{DType, Result as CandleResult, Tensor, Device, D};
use candle_nn::{VarBuilder, VarMap};
use candle_app::{LinearModel, Model};
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use base64::Engine;
use anyhow::Result;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use std::io::{self, Write};

struct Server {
    clients: HashMap<String, String>, // client_ip -> model_name
    ready_clients: HashMap<String, bool>, // client_ip -> is_ready
    models: HashMap<String, (LinearModel, VarMap, String)>,
    test_dataset: Option<candle_datasets::vision::Dataset>,
}

impl Server {
    fn new() -> Self {
        Server {
            clients: HashMap::new(),
            ready_clients: HashMap::new(),
            models: HashMap::new(),
            test_dataset: None,
        }
    }

    fn register(&mut self, client_ip: String, model: String) {
        println!("Registering client {} for model {}", client_ip, model);
        self.clients.insert(client_ip.clone(), model);
        self.ready_clients.insert(client_ip, false);
    }

    fn mark_ready(&mut self, client_ip: &str) {
        if let Some(ready) = self.ready_clients.get_mut(client_ip) {
            *ready = true;
            println!("Client {} marked as ready", client_ip);
        }
    }

    fn init(&mut self, model: String) -> CandleResult<()> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let global_model = LinearModel::new(vs)?;
        println!("Initializing model {}", model);
        self.models.insert(model, (global_model, varmap, "initialized".to_string()));
        if self.test_dataset.is_none() {
            self.test_dataset = Some(candle_datasets::vision::mnist::load()?);
        }
        Ok(())
    }

    fn get_model(&self, model_name: &str) -> Option<&(LinearModel, VarMap, String)> {
        self.models.get(model_name)
    }

    async fn aggregate_updates(
        &mut self,
        model_name: &str,
        updates: Vec<(Vec<f32>, Vec<f32>)>,
    ) -> CandleResult<()> {
        let (_, varmap, status) = self.models.get_mut(model_name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Model {} not found", model_name))
        })?;

        let mut weights_sum: Vec<f32> = vec![0.0; 10 * 784];
        let mut bias_sum: Vec<f32> = vec![0.0; 10];
        let num_clients = updates.len() as f32;

        for (weights_data, bias_data) in &updates {
            for (i, &w) in weights_data.iter().enumerate() {
                weights_sum[i] += w;
            }
            for (i, &b) in bias_data.iter().enumerate() {
                bias_sum[i] += b;
            }
        }

        let weights_avg: Vec<f32> = weights_sum.into_iter().map(|w| w / num_clients).collect();
        let bias_avg: Vec<f32> = bias_sum.into_iter().map(|b| b / num_clients).collect();

        let weights_tensor = Tensor::from_vec(weights_avg, &[10, 784], &Device::Cpu)?;
        let bias_tensor = Tensor::from_vec(bias_avg, &[10], &Device::Cpu)?;

        let mut data = varmap.data().lock().unwrap();
        data.get_mut("linear.weight")
            .expect("linear.weight missing")
            .set(&weights_tensor)?;
        data.get_mut("linear.bias")
            .expect("linear.bias missing")
            .set(&bias_tensor)?;

        *status = "ready".to_string();
        println!("Global model {} updated with {} client updates", model_name, updates.len());
        Ok(())
    }

    async fn train(&mut self, model_name: &str, rounds: usize, epochs: usize) -> Result<()> {
        let (tx, mut rx) = mpsc::channel(32);

        for round in 1..=rounds {
            println!("Starting training round {}", round);

            let ready_clients: Vec<String> = self.ready_clients
                .iter()
                .filter(|&(_, &ready)| ready)
                .map(|(ip, _)| ip.clone())
                .collect();
            if ready_clients.is_empty() {
                println!("No ready clients for round {}", round);
                sleep(Duration::from_secs(1)).await;
                continue;
            }

            let (weights_data, bias_data) = if let Some((model, _, _)) = self.get_model(model_name) {
                (
                    model.weight()?.to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>(),
                    model.bias()?.to_vec1::<f32>()?
                )
            } else {
                return Err(anyhow::anyhow!("Model {} not found", model_name));
            };
            let weights = bincode::serialize(&weights_data)?;
            let bias = bincode::serialize(&bias_data)?;

            let mut handles = Vec::new();
            for client_ip in &ready_clients {
                let tx = tx.clone();
                let model_name = model_name.to_string();
                let weights = weights.clone();
                let bias = bias.clone();
                let client_ip = client_ip.clone();
                let handle = tokio::spawn(async move {
                    match TcpStream::connect(&client_ip).await {
                        Ok(mut stream) => {
                            let train_message = format!(
                                "TRAIN|{}|{}|{}|{}",
                                model_name,
                                base64::engine::general_purpose::STANDARD.encode(&weights),
                                base64::engine::general_purpose::STANDARD.encode(&bias),
                                epochs
                            );
                            println!("Sending TRAIN to {} with {} epochs", client_ip, epochs);
                            stream.write_all(train_message.as_bytes()).await?;
                            stream.flush().await?;

                            let mut buffer = [0; 65536];
                            if let Ok(n) = stream.read(&mut buffer).await {
                                let response = String::from_utf8_lossy(&buffer[..n]);
                                if response.starts_with("UPDATE|") {
                                    let parts: Vec<&str> = response.split('|').collect();
                                    let weights_data: Vec<f32> = bincode::deserialize(&base64::engine::general_purpose::STANDARD.decode(parts[1])?)?;
                                    let bias_data: Vec<f32> = bincode::deserialize(&base64::engine::general_purpose::STANDARD.decode(parts[2])?)?;
                                    tx.send((weights_data, bias_data)).await?;
                                }
                            }
                        }
                        Err(e) => eprintln!("Failed to connect to {}: {}", client_ip, e),
                    }
                    Ok::<(), anyhow::Error>(())
                });
                handles.push(handle);
            }

            let mut updates = Vec::new();
            for _ in 0..ready_clients.len() {
                if let Some(update) = rx.recv().await {
                    updates.push(update);
                }
            }
            for handle in handles {
                handle.await??;
            }

            if !updates.is_empty() {
                self.aggregate_updates(model_name, updates).await?;
                println!("Completed training round {}", round);
            } else {
                println!("No updates received in round {}", round);
            }
            sleep(Duration::from_secs(1)).await;
        }
        Ok(())
    }

    fn test(&self, model_name: &str) -> CandleResult<f32> {
        let (model, _, _) = self.models.get(model_name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Model {} not found", model_name))
        })?;
        let test_dataset = self.test_dataset.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("Test dataset not loaded".into())
        })?;
        let dev = &Device::Cpu;
        let test_images = test_dataset.test_images.to_device(dev)?;
        let test_labels = test_dataset.test_labels.to_dtype(DType::U32)?.to_device(dev)?;
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

    async fn handle_client(stream: TcpStream, server: Arc<Mutex<Server>>) -> Result<()> {
        let mut buffer = [0; 65536];
        let peer_addr = stream.peer_addr()?.to_string();
        println!("Handling client connection from {}", peer_addr);
        let mut client_listening_addr: Option<String> = None;

        let mut stream = stream;

        loop {
            match stream.read(&mut buffer).await {
                Ok(0) => {
                    println!("Client {} disconnected", peer_addr);
                    break;
                }
                Ok(n) => {
                    let message = String::from_utf8_lossy(&buffer[..n]).to_string();
                    let parts: Vec<&str> = message.split('|').collect();

                    let mut server_guard = server.lock().await;
                    match parts[0] {
                        "REGISTER" if parts.len() == 3 => {
                            let client_ip = parts[1].to_string();
                            let model_name = parts[2].to_string();
                            server_guard.register(client_ip.clone(), model_name);
                            client_listening_addr = Some(client_ip.clone());
                            stream.write_all(b"Registered successfully").await?;
                            stream.flush().await?;
                        }
                        "READY" => {
                            if let Some(ref client_ip) = client_listening_addr {
                                server_guard.mark_ready(client_ip);
                                stream.write_all(b"Waiting for training round").await?;
                                stream.flush().await?;
                            } else {
                                stream.write_all(b"Error: Client not registered").await?;
                                stream.flush().await?;
                            }
                        }
                        "GET" if parts.len() == 2 => {
                            let model_name = parts[1];
                            if let Some((model, _, status)) = server_guard.get_model(model_name) {
                                let weights_data = model.weight()?.to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
                                let bias_data = model.bias()?.to_vec1::<f32>()?;
                                let weights = bincode::serialize(&weights_data)?;
                                let bias = bincode::serialize(&bias_data)?;
                                let response = format!(
                                    "MODEL|{}|{}|{}",
                                    base64::engine::general_purpose::STANDARD.encode(&weights),
                                    base64::engine::general_purpose::STANDARD.encode(&bias),
                                    status
                                );
                                stream.write_all(response.as_bytes()).await?;
                            } else {
                                stream.write_all(b"Model not found").await?;
                            }
                            stream.flush().await?;
                        }
                        "TEST" if parts.len() == 2 => {
                            let model_name = parts[1];
                            match server_guard.test(model_name) {
                                Ok(accuracy) => {
                                    let response = format!("ACCURACY|{}", accuracy);
                                    stream.write_all(response.as_bytes()).await?;
                                }
                                Err(e) => {
                                    stream.write_all(format!("Error: {}", e).as_bytes()).await?;
                                }
                            }
                            stream.flush().await?;
                        }
                        _ => {
                            stream.write_all(b"Invalid command").await?;
                            stream.flush().await?;
                        }
                    }
                    drop(server_guard);
                }
                Err(e) => {
                    eprintln!("Error reading from client {}: {}", peer_addr, e);
                    break;
                }
            }
        }
        Ok(())
    }

    fn handle_get_command(&self, model_name: &str) -> Result<()> {
        if let Some((model, _, status)) = self.get_model(model_name) {
            let weights_data = model.weight()?.to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
            let bias_data = model.bias()?.to_vec1::<f32>()?;
            println!("Model: {}", model_name);
            println!("Weights: {:?}", weights_data);
            println!("Bias: {:?}", bias_data);
            println!("Status: {}", status);
        } else {
            println!("Model '{}' not found", model_name);
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let server = Arc::new(Mutex::new(Server::new()));
    {
        let mut server_guard = server.lock().await;
        server_guard.init("mnist".to_string())?;
    }

    let listener = TcpListener::bind("127.0.0.1:50051").await?;
    println!("Server listening on 127.0.0.1:50051");
    println!("Type 'TRAIN <rounds> <epochs>' to start training, 'GET <model_name>' to retrieve model parameters and status, or 'exit' to quit");

    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    println!("New connection from {}", addr);
                    let server_clone_inner = Arc::clone(&server_clone);
                    tokio::spawn(async move {
                        if let Err(e) = Server::handle_client(stream, server_clone_inner).await {
                            eprintln!("Error handling client {}: {}", addr, e);
                        }
                    });
                }
                Err(e) => {
                    eprintln!("Error accepting connection: {}", e);
                    sleep(Duration::from_secs(1)).await;
                }
            }
        }
    });

    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("exit") {
            println!("Shutting down server...");
            break;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0].to_uppercase().as_str() {
            "TRAIN" if parts.len() == 3 => {
                let rounds = parts[1].parse::<usize>().map_err(|e| anyhow::anyhow!("Invalid rounds: {}", e))?;
                let epochs = parts[2].parse::<usize>().map_err(|e| anyhow::anyhow!("Invalid epochs: {}", e))?;
                let server_clone = Arc::clone(&server);
                tokio::spawn(async move {
                    let mut server_guard = server_clone.lock().await;
                    let ready_count = server_guard.ready_clients.values().filter(|&&ready| ready).count();
                    println!("Starting training with {} ready clients, {} rounds, {} epochs", ready_count, rounds, epochs);
                    if let Err(e) = server_guard.train("mnist", rounds, epochs).await {
                        eprintln!("Training error: {}", e);
                    }
                    if let Ok(accuracy) = server_guard.test("mnist") {
                        println!("Global model accuracy after training: {:.2}%", accuracy * 100.0);
                    }
                    let client_ips: Vec<String> = server_guard.clients.keys().cloned().collect();
                    drop(server_guard);
                    for client_ip in client_ips {
                        if let Ok(mut stream) = TcpStream::connect(&client_ip).await {
                            stream.write_all(b"COMPLETE").await?;
                            stream.flush().await?;
                        } else {
                            eprintln!("Failed to notify client {}", client_ip);
                        }
                    }
                    Ok::<(), anyhow::Error>(())
                });
            }
            "GET" if parts.len() == 2 => {
                let model_name = parts[1];
                let server_guard = server.lock().await;
                server_guard.handle_get_command(model_name)?;
            }
            _ => {
                println!("Invalid command. Use 'TRAIN <rounds> <epochs>', 'GET <model_name>', or 'exit'");
            }
        }
    }

    Ok(())
}