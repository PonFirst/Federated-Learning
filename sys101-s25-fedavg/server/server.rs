use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use candle_core::{DType, Result};
use candle_nn::{VarBuilder, VarMap};
use candle_app::LinearModel;
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct Server {
    clients: HashMap<String, String>,
    models: HashMap<String, LinearModel>,
}

impl Server {
    fn new() -> Self {
        Server {
            clients: HashMap::new(),
            models: HashMap::new(),
        }
    }

    fn register(&mut self, client_ip: String, model: String) {
        println!("Registering client {} for model {}", client_ip, model);
        self.clients.insert(client_ip, model);
    }

    fn init(&mut self, model: String) -> Result<()> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &candle_core::Device::Cpu);
        let global_model = LinearModel::new(vs)?;
        println!("Initializing model {}", model);
        self.models.insert(model, global_model);
        Ok(())
    }

    fn train(&mut self, model_name: String, rounds: usize) -> Result<()> {
        println!("Training {} for {} rounds (placeholder)", model_name, rounds);
        Ok(())
    }

    async fn handle_client(&mut self, mut stream: tokio::net::TcpStream) -> Result<()> {
        let mut buffer = [0; 1024];
        let n = stream.read(&mut buffer).await?;
        let message = String::from_utf8_lossy(&buffer[..n]);
        let parts: Vec<&str> = message.split('|').collect();
        if parts.len() == 2 && parts[0] == "REGISTER" {
            let client_ip = parts[1].to_string();
            let model_name = "mnist".to_string();
            self.register(client_ip.clone(), model_name);
            stream.write_all(b"Registered successfully").await?;
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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