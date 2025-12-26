pub struct FastCompressor {
    base: f64,
    counts: Vec<u64>,
}

impl FastCompressor {
    pub fn new(base: f64) -> Self {
        assert!(base > 1.0, "底数必须大于 1.0");
        Self {
            base,
            counts: Vec::new(),
        }
    }
    pub fn compress(&mut self, data: Vec<u64>) {
        for value in data {
            let bucket_index = self.get_bucket_index(value);
            if bucket_index >= self.counts.len() {
                self.counts.resize(bucket_index + 1, 0);
            }
            self.counts[bucket_index] += 1;
        }
    }

    fn get_bucket_index(&self, value: u64) -> usize {
        if value == 0 {
            return 0;
        }

        let value_f64 = value as f64;
        let mut index = 0;
        let mut upper_bound = self.base;

        while value_f64 > upper_bound {
            index += 1;
            upper_bound *= self.base;
        }

        index
    }

    pub fn get_result(&self) -> (usize, f64, Vec<u64>) {
        (self.counts.len(), self.base, self.counts.clone())
    }

    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn base(&self) -> f64 {
        self.base
    }

    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    pub fn reset(&mut self) {
        self.counts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_compressor_base_2() {
        let mut compressor = FastCompressor::new(2.0);
        compressor.compress(vec![10, 8, 5, 3, 2, 1]);

        let (len, base, counts) = compressor.get_result();
        assert_eq!(len, 4);
        assert_eq!(base, 2.0);
        assert_eq!(counts, vec![2, 1, 2, 1]);
    }

    #[test]
    fn test_fast_compressor_base_1_5() {
        let mut compressor = FastCompressor::new(1.5);
        compressor.compress(vec![5, 3, 2, 1]);

        let (len, base, counts) = compressor.get_result();
        assert_eq!(len, 4);
        assert_eq!(base, 1.5);
        assert_eq!(counts.len(), 4);
    }

    #[test]
    fn test_fast_compressor_empty() {
        let mut compressor = FastCompressor::new(2.0);
        compressor.compress(vec![]);

        let (len, _, counts) = compressor.get_result();
        assert_eq!(len, 0);
        assert_eq!(counts, Vec::<u64>::new());
    }

    #[test]
    fn test_fast_compressor_zero() {
        let mut compressor = FastCompressor::new(2.0);
        compressor.compress(vec![0, 0, 0]);

        let (_, _, counts) = compressor.get_result();
        assert_eq!(counts[0], 3);
    }

    #[test]
    fn test_fast_compressor_multiple_calls() {
        let mut compressor = FastCompressor::new(2.0);
        compressor.compress(vec![1, 2]);
        compressor.compress(vec![3, 4, 5]);

        let (_, _, counts) = compressor.get_result();
        assert_eq!(counts, vec![2, 2, 1]);
    }

    #[test]
    fn test_reset() {
        let mut compressor = FastCompressor::new(2.0);
        compressor.compress(vec![1, 2, 3]);
        compressor.reset();

        let (len, _, counts) = compressor.get_result();
        assert_eq!(len, 0);
        assert_eq!(counts, Vec::<u64>::new());
    }
}
