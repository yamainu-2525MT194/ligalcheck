function updateDatasetCount() {
    fetch('/get_dataset_count')
        .then(response => response.json())
        .then(data => {
            const datasetCountElement = document.getElementById('dataset-count');
            if (datasetCountElement) {
                datasetCountElement.textContent = data.count;
            }
        })
        .catch(error => {
            console.error('Error fetching dataset count:', error);
        });
}

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('contract-upload-form');
    const uploadStatus = document.getElementById('upload-status');

    // Initial dataset count update
    updateDatasetCount();

    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(uploadForm);
            
            fetch('/upload_contract', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    uploadStatus.textContent = 'Contract uploaded successfully!';
                    uploadStatus.style.color = 'green';
                    
                    // Update dataset count after successful upload
                    updateDatasetCount();
                    
                    // Optional: Reset form after successful upload
                    uploadForm.reset();
                } else {
                    uploadStatus.textContent = data.message || 'Upload failed';
                    uploadStatus.style.color = 'red';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                uploadStatus.textContent = 'An error occurred during upload';
                uploadStatus.style.color = 'red';
            });
        });
    }
});
