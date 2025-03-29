use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use candle_core::{DType, Result as CandleResult, Tensor, Device, D};
use candle_nn::{VarBuilder, VarMap};
use candle_app::{LinearModel, Model};
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt, AsyncBufReadExt, BufReader};
use base64::Engine;
use anyhow::Result;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use std::io::{self, Write};

const MODEL_NAME: &str = "mnist";

struct Server {
    clients: HashMap<String, String>, // client_ip -> model_name
    ready_clients: HashMap<String, bool>, // client_ip -> is_ready
    model: Option<(LinearModel, VarMap, String)>, // (model, varmap, status)
    test_dataset: Option<candle_datasets::vision::Dataset>,
}

impl Server {
    fn new() -> Self {  // Fixed typo from fnageddonnew to fn new
        Server {
            clients: HashMap::new(),
            ready_clients: HashMap::new(),
            model: None,
            test_dataset: None,
        }
    }

    fn register(&mut self, client_ip: String) {
        println!("Registering client {} for model {}", client_ip, MODEL_NAME);
        self.clients.insert(client_ip.clone(), MODEL_NAME.to_string());
        self.ready_clients.insert(client_ip, false);
    }

    fn mark_ready(&mut self, client_ip: &str) {
        if let Some(ready) = self.ready_clients.get_mut(client_ip) {
            *ready = true;
            println!("Client {} marked as ready", client_ip);
        }
    }

    fn remove_client(&mut self, client_ip: &str) {
        println!("Removing client {} from tracking", client_ip);
        self.clients.remove(client_ip);
        self.ready_clients.remove(client_ip);
    }

    fn init(&mut self) -> CandleResult<()> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let global_model = LinearModel::new(vs)?;
        println!("Initializing model {}", MODEL_NAME);
        self.model = Some((global_model, varmap, "initialized".to_string()));
        if self.test_dataset.is_none() {
            self.test_dataset = Some(candle_datasets::vision::mnist::load()?);
        }
        Ok(())
    }

    fn get_model(&self) -> Option<&(LinearModel, VarMap, String)> {
        self.model.as_ref()
    }

    async fn aggregate_updates(&mut self, updates: Vec<(Vec<f32>, Vec<f32>)>) -> CandleResult<()> {
        let (model, varmap, status) = self.model.as_mut().ok_or_else(|| {
            candle_core::Error::Msg("Model not initialized".to_string())
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
        println!("Global model {} updated with {} client updates", MODEL_NAME, updates.len());
        Ok(())
    }

    async fn train(&self, rounds: usize, epochs: usize, server: Arc<Mutex<Self>>) -> Result<()> {
        let (tx, mut rx) = mpsc::channel(32);
        {
            let mut server_guard = server.lock().await;
            if let Some((_, _, status)) = &mut server_guard.model {
                *status = "training".to_string();
            }
        }

        for round in 1..=rounds {
            println!("Starting training round {}", round);

            let ready_clients: Vec<String> = {
                let server_guard = server.lock().await;
                server_guard.ready_clients
                    .iter()
                    .filter(|&(_, &ready)| ready)
                    .map(|(ip, _)| ip.clone())
                    .collect()
            };
            if ready_clients.is_empty() {
                println!("No ready clients for round {}", round);
                sleep(Duration::from_secs(1)).await;
                continue;
            }

            let (weights_data, bias_data) = {
                let server_guard = server.lock().await;
                if let Some((model, _, _)) = server_guard.get_model() {
                    (
                        model.weight()?.to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>(),
                        model.bias()?.to_vec1::<f32>()?
                    )
                } else {
                    return Err(anyhow::anyhow!("Model not initialized"));
                }
            };
            let weights = bincode::serialize(&weights_data)?;
            let bias = bincode::serialize(&bias_data)?;

            let mut handles = Vec::new();
            for client_ip in &ready_clients {
                let tx = tx.clone();
                let weights = weights.clone();
                let bias = bias.clone();
                let client_ip = client_ip.clone();
                let handle = tokio::spawn(async move {
                    match TcpStream::connect(&client_ip).await {
                        Ok(mut stream) => {
                            let train_message = format!(
                                "TRAIN|{}|{}|{}|{}",
                                MODEL_NAME,
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
                                    let weights_data: Vec<f32> = bincode::deserialize(
                                        &base64::engine::general_purpose::STANDARD.decode(parts[1])?,
                                    )?;
                                    let bias_data: Vec<f32> = bincode::deserialize(
                                        &base64::engine::general_purpose::STANDARD.decode(parts[2])?,
                                    )?;
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
                let mut server_guard = server.lock().await;
                server_guard.aggregate_updates(updates).await?;
                println!("Completed training round {}", round);
            } else {
                println!("No updates received in round {}", round);
            }
            sleep(Duration::from_secs(1)).await;
        }

        let client_ips = {
            let server_guard = server.lock().await;
            server_guard.clients.keys().cloned().collect::<Vec<String>>()
        };
        for client_ip in client_ips {
            if let Ok(mut stream) = TcpStream::connect(&client_ip).await {
                stream.write_all(b"COMPLETE").await?;
                stream.flush().await?;
            } else {
                eprintln!("Failed to notify client {}", client_ip);
            }
        }

        Ok(())
    }

    fn test(&self) -> CandleResult<f32> {
        let (model, _, _) = self.model.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("Model not initialized".to_string())
        })?;
        let test_dataset = self.test_dataset.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("Test dataset not loaded".to_string())
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
                    if let Some(ref client_ip) = client_listening_addr {
                        let mut server_guard = server.lock().await;
                        server_guard.remove_client(client_ip);
                    }
                    break;
                }
                Ok(n) => {
                    let message = String::from_utf8_lossy(&buffer[..n]).to_string();
                    let parts: Vec<&str> = message.split('|').collect();

                    let mut server_guard = server.lock().await;
                    match parts[0] {
                        "REGISTER" if parts.len() == 2 => {
                            let client_ip = parts[1].to_string();
                            server_guard.register(client_ip.clone());
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
                        "GET" => {
                            if let Some((model, _, status)) = server_guard.get_model() {
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
                        "TEST" => {
                            match server_guard.test() {
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
                    if let Some(ref client_ip) = client_listening_addr {
                        let mut server_guard = server.lock().await;
                        server_guard.remove_client(client_ip);
                    }
                    break;
                }
            }
        }
        Ok(())
    }

    fn handle_get_command(&self) -> Result<()> {
        if let Some((model, _, status)) = self.get_model() {
            let weights_data = model.weight()?.to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
            let bias_data = model.bias()?.to_vec1::<f32>()?;
            println!("Model: {}", MODEL_NAME);
            println!("Weights: {:?}", weights_data);
            println!("Bias: {:?}", bias_data);
            println!("Status: {}", status);
        } else {
            println!("Model '{}' not found", MODEL_NAME);
        }
        Ok(())
    }

    async fn handle_commands(server: Arc<Mutex<Server>>) -> Result<()> {
        println!("Type 'INIT', 'TRAIN <rounds> <epochs>', 'GET', 'TEST', or 'exit'");
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut input = String::new();

        loop {
            print!("> ");
            io::stdout().flush()?;
            input.clear();
            reader.read_line(&mut input).await?;
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
                "INIT" => {
                    let mut server_guard = server.lock().await;
                    if let Err(e) = server_guard.init() {
                        eprintln!("Init error: {}", e);
                    }
                }
                "TRAIN" if parts.len() == 3 => {
                    let rounds = match parts[1].parse::<usize>() {
                        Ok(r) => r,
                        Err(e) => {
                            println!("Invalid rounds: {}", e);
                            continue;
                        }
                    };
                    let epochs = match parts[2].parse::<usize>() {
                        Ok(e) => e,
                        Err(e) => {
                            println!("Invalid epochs: {}", e);
                            continue;
                        }
                    };
                    let server_clone = Arc::clone(&server);
                    let server_clone_for_train = Arc::clone(&server_clone);
                    tokio::spawn(async move {
                        if let Err(e) = Server::train(&Server::new(), rounds, epochs, server_clone_for_train).await {
                            eprintln!("Training error: {}", e);
                        } else {
                            let server_guard = server_clone.lock().await;
                            if let Ok(accuracy) = server_guard.test() {
                                println!("Global model accuracy after training: {:.2}%", accuracy * 100.0);
                            }
                        }
                    });
                    println!("Training command issued with {} rounds and {} epochs", rounds, epochs);
                }
                "GET" => {
                    let server_guard = server.lock().await;
                    server_guard.handle_get_command()?;
                }
                "TEST" => {
                    let server_guard = server.lock().await;
                    match server_guard.test() {
                        Ok(accuracy) => println!("Accuracy: {:.2}%", accuracy * 100.0),
                        Err(e) => eprintln!("Test error: {}", e),
                    }
                }
                _ => {
                    println!("Invalid command. Use 'INIT', 'TRAIN <rounds> <epochs>', 'GET', 'TEST', or 'exit'");
                }
            }
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let server = Arc::new(Mutex::new(Server::new()));
    let listener = TcpListener::bind("127.0.0.1:50051").await?;
    println!("Server listening on 127.0.0.1:50051");

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

    Server::handle_commands(server).await?;
    Ok(())
}