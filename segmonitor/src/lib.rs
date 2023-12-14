use std::process::Command;
use serde::{Deserialize, Serialize};
use mongodb::{Client, options::ClientOptions};

#[derive(Debug, Deserialize, Serialize)]
struct Config {
    model_type: String,
    iterations: i32,
    warmup: i32,
    raw_file: String,
    voxel_size: i32,
    python_script_path: String,
}

pub mod segmonitor {
    pub fn train_model_from_config(config_path: &str) {
        let config = load_config(config_path);

        println!("Training model: {}", config.model_type);
        println!("Iterations: {}", config.iterations);
        println!("Warmup: {}", config.warmup);
        println!("Raw file: {}", config.raw_file);
        println!("Voxel size: {}", config.voxel_size);

        call_python_train(&config.python_script_path);

        save_to_mongodb(&config);
    }

    fn load_config(config_path: &str) -> Config {
        let config_str = std::fs::read_to_string(config_path).expect("Error reading config file");
        serde_json::from_str(&config_str).expect("Error parsing JSON")
    }

    fn call_python_train(script_path: &str) {
        let output = Command::new("python")
            .arg(script_path)
            .output()
            .expect("Failed to execute training");

        if output.status.success() {
            println!("Traning executed successfully!");
        } else {
            println!("Error executing training:\n{}", String::from_utf8_lossy(&output.stderr));
        }
    }

    fn save_to_mongodb(config: &Config) {
        println!("Saving metrics to MongoDB...");
        let client_options = ClientOptions::parse("mongodb://localhost:27017").unwrap();
        let client = Client::with_options(client_options).unwrap();
        // TODO: dd MongoDB insertion logic here
    }
}
