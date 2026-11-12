use crate::diskio::aligned_reader::AlignedReader;
use crate::diskio::file::SharedFd;
use crate::diskio::file::file_size_fd;
use crate::{IoStatsTracker, SortInput};
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

// Default GenSort record layout: 10-byte key, 90-byte payload
pub const GENSORT_KEY_SIZE: usize = 10;
pub const GENSORT_PAYLOAD_SIZE: usize = 90;

/// Direct I/O reader for fixed-size binary records (GenSort format by default).
///
/// Each record is `key_size` key bytes followed immediately by `value_size`
/// value bytes, with no delimiters or length prefixes.
#[derive(Clone)]
pub struct GenSortInputDirect {
    fd: Arc<SharedFd>,
    num_records: usize,
    key_size: usize,
    value_size: usize,
}

impl GenSortInputDirect {
    /// Open a standard GenSort file (10-byte keys, 90-byte payloads).
    pub fn new(path: impl AsRef<Path>) -> Result<Self, String> {
        Self::with_record_layout(path, GENSORT_KEY_SIZE, GENSORT_PAYLOAD_SIZE)
    }

    /// Open a file of fixed-size records with the given key and value sizes.
    pub fn with_record_layout(
        path: impl AsRef<Path>,
        key_size: usize,
        value_size: usize,
    ) -> Result<Self, String> {
        if key_size == 0 {
            return Err("Key size must be greater than 0".to_string());
        }
        let record_size = key_size + value_size;

        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(format!("File does not exist: {:?}", path));
        }

        let fd = Arc::new(SharedFd::new_from_path(&path, false).map_err(|e| {
            format!(
                "Failed to open file with Direct I/O: {}: {}",
                path.display(),
                e
            )
        })?);

        let file_size = file_size_fd(fd.as_raw_fd())
            .map_err(|e| format!("Failed to get file size for {:?}: {}", path, e))?;

        // Verify file size is a multiple of record size
        if file_size % record_size as u64 != 0 {
            return Err(format!(
                "File size {} is not a multiple of record size {}",
                file_size, record_size
            ));
        }

        let num_records = (file_size / record_size as u64) as usize;

        Ok(Self {
            fd,
            num_records,
            key_size,
            value_size,
        })
    }

    pub fn key_size(&self) -> usize {
        self.key_size
    }

    pub fn value_size(&self) -> usize {
        self.value_size
    }

    pub fn record_size(&self) -> usize {
        self.key_size + self.value_size
    }

    pub fn len(&self) -> usize {
        self.num_records
    }

    pub fn file_size(&self) -> Result<u64, String> {
        file_size_fd(self.fd.as_raw_fd()).map_err(|e| format!("Failed to get file size: {}", e))
    }

    pub fn is_empty(&self) -> bool {
        self.num_records == 0
    }
}

impl SortInput for GenSortInputDirect {
    fn create_parallel_scanners(
        &self,
        num_scanners: usize,
        io_tracker: Option<IoStatsTracker>,
    ) -> Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>> {
        if self.num_records == 0 {
            return vec![];
        }

        let records_per_scanner = self.num_records.div_ceil(num_scanners);
        let mut scanners = Vec::new();

        for scanner_id in 0..num_scanners {
            let start_record = scanner_id * records_per_scanner;
            if start_record >= self.num_records {
                break;
            }

            let end_record = ((scanner_id + 1) * records_per_scanner).min(self.num_records);
            let scanner = GenSortScanner::new(
                self.fd.clone(),
                start_record,
                end_record,
                self.key_size,
                self.value_size,
                io_tracker.clone(),
            );

            match scanner {
                Ok(s) => scanners
                    .push(Box::new(s) as Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>),
                Err(e) => {
                    eprintln!("Failed to create scanner {}: {}", scanner_id, e);
                    // Continue with other scanners
                }
            }
        }

        scanners
    }

    fn estimated_size_bytes(&self) -> Option<u64> {
        self.file_size().ok()
    }
}

/// Scanner for reading a range of records from a GenSort file
struct GenSortScanner {
    reader: AlignedReader,
    current_record: usize,
    end_record: usize,
    key_size: usize,
    value_size: usize,
}

impl GenSortScanner {
    fn new(
        fd: Arc<SharedFd>,
        start_record: usize,
        end_record: usize,
        key_size: usize,
        value_size: usize,
        io_tracker: Option<IoStatsTracker>,
    ) -> Result<Self, String> {
        // Open file with Direct I/O
        // Create aligned reader with optional IO tracking
        let mut reader = if let Some(tracker) = io_tracker {
            AlignedReader::from_fd_with_tracer(fd, Some(tracker))
                .map_err(|e| format!("Failed to create aligned reader: {}", e))?
        } else {
            AlignedReader::from_fd(fd)
                .map_err(|e| format!("Failed to create aligned reader: {}", e))?
        };

        // Seek to start position (must be aligned)
        reader
            .seek((start_record * (key_size + value_size)) as u64)
            .map_err(|e| format!("Failed to seek: {}", e))?;

        Ok(Self {
            reader,
            current_record: start_record,
            end_record,
            key_size,
            value_size,
        })
    }

    fn read_record(&mut self) -> Option<(Vec<u8>, Vec<u8>)> {
        if self.current_record >= self.end_record {
            return None;
        }

        let mut key = vec![0u8; self.key_size];
        let mut payload = vec![0u8; self.value_size];

        // Read the next record into the buffer if needed

        self.reader
            .read_exact(&mut key)
            .map_err(|e| eprintln!("Failed to read key: {}", e))
            .ok()?;
        self.reader
            .read_exact(&mut payload)
            .map_err(|e| eprintln!("Failed to read payload: {}", e))
            .ok()?;

        self.current_record += 1;

        Some((key, payload))
    }
}

impl Iterator for GenSortScanner {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        self.read_record()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    const KEY_SIZE: usize = GENSORT_KEY_SIZE;
    const PAYLOAD_SIZE: usize = GENSORT_PAYLOAD_SIZE;
    const RECORD_SIZE: usize = KEY_SIZE + PAYLOAD_SIZE;

    fn generate_gensort_data(num_records: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(num_records * RECORD_SIZE);

        for i in 0..num_records {
            // Generate 10-byte key
            let key = format!("key{:06}", i);
            let mut key_bytes = key.as_bytes().to_vec();
            key_bytes.resize(KEY_SIZE, 0); // Pad with zeros if needed

            // Generate 90-byte payload
            let payload = format!("payload{:03}", i);
            let mut payload_bytes = payload.as_bytes().to_vec();
            payload_bytes.resize(PAYLOAD_SIZE, b'x'); // Pad with 'x' characters

            data.extend_from_slice(&key_bytes);
            data.extend_from_slice(&payload_bytes);
        }

        data
    }

    fn create_test_file(num_records: usize) -> tempfile::TempPath {
        let data = generate_gensort_data(num_records);

        // Use NamedTempFile so we can return a TempPath that keeps
        // the file alive for the duration of the test without
        // requiring Debug on GenSortInputDirect.
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&data).unwrap();
        tmp.as_file_mut().sync_all().unwrap();
        tmp.into_temp_path()
    }

    #[test]
    fn test_gensort_input_direct_creation() {
        let temp_file = create_test_file(10);

        let input = GenSortInputDirect::new(&temp_file).unwrap();
        assert_eq!(input.len(), 10);
        assert!(!input.is_empty());

        let file_size = input.file_size().unwrap();
        assert_eq!(file_size, (10 * RECORD_SIZE) as u64);
    }

    #[test]
    fn test_gensort_input_invalid_file_size() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("invalid_gensort.data");

        // Create file with invalid size (not multiple of RECORD_SIZE)
        let mut file = File::create(&file_path).unwrap();
        file.write_all(&vec![0u8; 50]).unwrap(); // 50 bytes, not multiple of 100
        file.sync_all().unwrap();

        let result = GenSortInputDirect::new(&file_path);
        assert!(result.is_err());
        match result {
            Err(e) => assert!(e.contains("not a multiple of record size")),
            Ok(_) => panic!("expected error for invalid file size"),
        }
    }

    #[test]
    fn test_single_scanner_reads_all_records() {
        let temp_file = create_test_file(5);
        let input = GenSortInputDirect::new(&temp_file).unwrap();

        let scanners = input.create_parallel_scanners(1, None);
        assert_eq!(scanners.len(), 1);

        let mut scanner = scanners.into_iter().next().unwrap();
        let mut count = 0;
        let mut records = Vec::new();

        while let Some((key, payload)) = scanner.next() {
            assert_eq!(key.len(), KEY_SIZE);
            assert_eq!(payload.len(), PAYLOAD_SIZE);
            records.push((key, payload));
            count += 1;
        }

        assert_eq!(count, 5);

        // Verify first record
        let expected_key = b"key000000\0";
        let mut expected_payload = b"payload000".to_vec();
        expected_payload.resize(PAYLOAD_SIZE, b'x');

        assert_eq!(&records[0].0, expected_key);
        assert_eq!(&records[0].1, &expected_payload);
    }

    #[test]
    fn test_multiple_scanners_read_all_records() {
        let temp_file = create_test_file(10);
        let input = GenSortInputDirect::new(&temp_file).unwrap();

        let scanners = input.create_parallel_scanners(3, None);
        assert!(scanners.len() <= 3); // May be fewer if not enough records

        let mut total_count = 0;
        for mut scanner in scanners {
            while let Some((key, payload)) = scanner.next() {
                assert_eq!(key.len(), KEY_SIZE);
                assert_eq!(payload.len(), PAYLOAD_SIZE);
                total_count += 1;
            }
        }

        assert_eq!(total_count, 10);
    }

    #[test]
    fn test_empty_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("empty_gensort.data");
        File::create(&file_path).unwrap();

        let input = GenSortInputDirect::new(&file_path).unwrap();
        assert_eq!(input.len(), 0);
        assert!(input.is_empty());

        let scanners = input.create_parallel_scanners(2, None);
        assert_eq!(scanners.len(), 0);
    }

    #[test]
    fn test_record_content_correctness() {
        let temp_file = create_test_file(3);
        let input = GenSortInputDirect::new(&temp_file).unwrap();

        let mut scanner = input
            .create_parallel_scanners(1, None)
            .into_iter()
            .next()
            .unwrap();

        // Test first record
        let (key, payload) = scanner.next().unwrap();
        let key_str = std::str::from_utf8(&key[..9]).unwrap();
        assert!(key_str.starts_with("key"));
        assert_eq!(&key_str[3..], "000000");
        assert_eq!(&payload[..10], b"payload000");

        // Test second record
        let (key, payload) = scanner.next().unwrap();
        let key_str = std::str::from_utf8(&key[..9]).unwrap();
        assert!(key_str.starts_with("key"));
        assert_eq!(&key_str[3..], "000001");
        assert_eq!(&payload[..10], b"payload001");

        // Test third record
        let (key, payload) = scanner.next().unwrap();
        let key_str = std::str::from_utf8(&key[..9]).unwrap();
        assert!(key_str.starts_with("key"));
        assert_eq!(&key_str[3..], "000002");
        assert_eq!(&payload[..10], b"payload002");

        // Should be no more records
        assert!(scanner.next().is_none());
    }

    /// Write packed 8-byte-key / 8-byte-value records (big-endian u64s).
    fn create_8_8_file(keys: &[u64]) -> tempfile::TempPath {
        let mut data = Vec::with_capacity(keys.len() * 16);
        for &k in keys {
            data.extend_from_slice(&k.to_be_bytes());
            data.extend_from_slice(&(!k).to_be_bytes());
        }
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&data).unwrap();
        tmp.as_file_mut().sync_all().unwrap();
        tmp.into_temp_path()
    }

    #[test]
    fn test_8_8_layout_reads_all_records() {
        // Enough records that scanner boundaries fall at non-block-aligned offsets.
        let keys: Vec<u64> = (0..10_007u64)
            .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        let temp_file = create_8_8_file(&keys);
        let input = GenSortInputDirect::with_record_layout(&temp_file, 8, 8).unwrap();
        assert_eq!(input.len(), keys.len());
        assert_eq!(input.record_size(), 16);

        for num_scanners in [1, 3, 7] {
            let mut got = Vec::new();
            for scanner in input.create_parallel_scanners(num_scanners, None) {
                for (key, value) in scanner {
                    assert_eq!(key.len(), 8);
                    assert_eq!(value.len(), 8);
                    let k = u64::from_be_bytes(key.try_into().unwrap());
                    let v = u64::from_be_bytes(value.try_into().unwrap());
                    assert_eq!(v, !k);
                    got.push(k);
                }
            }
            assert_eq!(got, keys, "mismatch with {} scanners", num_scanners);
        }
    }

    #[test]
    fn test_8_8_layout_invalid_file_size() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&[0u8; 20]).unwrap(); // not a multiple of 16
        tmp.as_file_mut().sync_all().unwrap();
        let path = tmp.into_temp_path();
        match GenSortInputDirect::with_record_layout(&path, 8, 8) {
            Err(e) => assert!(e.contains("not a multiple of record size 16")),
            Ok(_) => panic!("expected error for invalid file size"),
        }
    }

    #[test]
    fn test_zero_key_size_rejected() {
        let temp_file = create_8_8_file(&[1, 2, 3]);
        assert!(GenSortInputDirect::with_record_layout(&temp_file, 0, 16).is_err());
    }
}
