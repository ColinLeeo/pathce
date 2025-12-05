use std::fmt;

#[derive(Debug)]
pub enum GCardError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Bincode(bincode::Error),
    InvalidData(String),
}

impl fmt::Display for GCardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GCardError::Io(e) => write!(f, "IO error: {}", e),
            GCardError::Json(e) => write!(f, "JSON error: {}", e),
            GCardError::Bincode(e) => write!(f, "Bincode error: {}", e),
            GCardError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
        }
    }
}

impl std::error::Error for GCardError {}

impl From<std::io::Error> for GCardError {
    fn from(err: std::io::Error) -> Self {
        GCardError::Io(err)
    }
}

impl From<serde_json::Error> for GCardError {
    fn from(err: serde_json::Error) -> Self {
        GCardError::Json(err)
    }
}

impl From<bincode::Error> for GCardError {
    fn from(err: bincode::Error) -> Self {
        GCardError::Bincode(err)
    }
}

pub type GCardResult<T> = Result<T, GCardError>;

