async function uploadImage() {
    const fileInput = document.getElementById("imageInput");
    const resultText = document.getElementById("result");
    const spinner = document.getElementById("spinner");
  
    resultText.textContent = "";
    resultText.className = "result-text";
    
    if (fileInput.files.length === 0) {
      resultText.textContent = "Please select an image.";
      return;
    }
  
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
  
    spinner.classList.remove("hidden");
  
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
      });
  
      const data = await response.json();
      spinner.classList.add("hidden");
  
      resultText.textContent = "Prediction: " + data.result;
      resultText.classList.add(data.result === "Tumor" ? "tumor" : "no-tumor");
  
    } catch (error) {
      spinner.classList.add("hidden");
      resultText.textContent = "Error: " + error.message;
      resultText.style.color = "red";
    }
  }
  