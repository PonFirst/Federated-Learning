use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use candle_core::{DType, Result as CandleResult, Tensor, Device};
use candle_nn::{VarBuilder, VarMap};
use candle_app::{LinearModel, Model};
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use base64::Engine;
use anyhow::Result;

struct Server {
    clients: HashMap<String, String>,
    models: HashMap<String, (LinearModel, VarMap, String)>,
    test_dataset: Option<candle_datasets::vision::Dataset>,
}

impl Server {
    fn new() -> Self {
        Server {
            clients: HashMap::new(),
            models: HashMap::new(),
            test_dataset: None,
        }
    }

    fn register(&mut self, client_ip: String, model: String) {
        println!("Registering client {} for model {}", client_ip, model);
        self.clients.insert(client_ip, model);
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
        let (model, varmap, _) = self.models.get_mut(model_name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Model {} not found", model_name))
        })?;

        let mut weights_sum: Vec<f32> = vec![0.0; 10 * 784];
        let mut bias_sum: Vec<f32> = vec![0.0; 10];
        let num_clients = updates.len() as f32;

        for (weights_data, bias_data) in updates {
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

        Ok(())
    }

    async fn handle_client(&mut self, mut stream: TcpStream) -> Result<()> {
        let mut buffer = [0; 65536];
        loop {
            match stream.read(&mut buffer).await {
                Ok(0) => {
                    println!("Client disconnected");
                    break;
                }
                Ok(n) => {
                    let message = String::from_utf8_lossy(&buffer[..n]);
                    let parts: Vec<&str> = message.split('|').collect();
                    match parts[0] {
                        "REGISTER" if parts.len() == 3 => {
                            let client_ip = parts[1].to_string();
                            let model_name = parts[2].to_string();
                            self.register(client_ip, model_name);
                            stream.write_all(b"Registered successfully").await?;
                        }
                        "GET" if parts.len() == 2 => {
                            let model_name = parts[1];
                            if let Some((model, _, status)) = self.get_model(model_name) {
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
                        }
                        "UPDATE" if parts.len() == 3 => {
                            let weights_bytes = base64::engine::general_purpose::STANDARD.decode(parts[1])?;
                            let bias_bytes = base64::engine::general_purpose::STANDARD.decode(parts[2])?;
                            let weights_data: Vec<f32> = bincode::deserialize(&weights_bytes)?;
                            let bias_data: Vec<f32> = bincode::deserialize(&bias_bytes)?;
                            println!(
                                "Received update: weights len={}, bias len={}",
                                weights_data.len(), bias_data.len()
                            );
                            // Perform aggregation
                            match self.aggregate_updates("mnist", vec![(weights_data, bias_data)]).await {
                                Ok(()) => {
                                    println!("Update successfully aggregated into global model");
                                    stream.write_all(b"Update received").await?;
                                }
                                Err(e) => {
                                    eprintln!("Failed to aggregate update: {}", e);
                                    stream.write_all(b"Update failed").await?;
                                }
                            }
                        }
                        _ => {
                            stream.write_all(b"Invalid command").await?;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error reading from client: {}", e);
                    break;
                }
            }
            stream.flush().await?;
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

    loop {
        let (stream, addr) = listener.accept().await?;
        println!("New connection from {}", addr);
        let server_clone = Arc::clone(&server);
        tokio::spawn(async move {
            let mut server_guard = server_clone.lock().await;
            if let Err(e) = server_guard.handle_client(stream).await {
                eprintln!("Error handling client: {}", e);
            }
        });
    }
}