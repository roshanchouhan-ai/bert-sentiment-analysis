const API_BASE_URL = "";


// =========================
// Get HTML Elements
// =========================

const reviewInput = document.getElementById("review");
const analyzeButton = document.getElementById("analyze-btn");
const csvButton = document.getElementById("csv-btn");
const csvFileInput = document.getElementById("csv-file");

const characterCount = document.getElementById("character-count");

const resultSection = document.getElementById("result-section");
const sentimentElement = document.getElementById("sentiment");
const confidenceElement = document.getElementById("confidence");

const errorMessage = document.getElementById("error-message");


// =========================
// Character Counter
// =========================

reviewInput.addEventListener("input", () => {

    const count = reviewInput.value.length;

    characterCount.textContent = `${count} characters`;

});


// =========================
// Show Error
// =========================

function showError(message) {

    errorMessage.textContent = message;
    errorMessage.hidden = false;

}


// =========================
// Clear Error
// =========================

function clearError() {

    errorMessage.textContent = "";
    errorMessage.hidden = true;

}


// =========================
// Show Result
// =========================

function showResult(sentiment, confidence) {

    sentimentElement.textContent = sentiment;

    confidenceElement.textContent =
        `${(confidence * 100).toFixed(2)}%`;

    resultSection.hidden = false;

}


// =========================
// Analyze Review
// =========================

analyzeButton.addEventListener("click", async () => {

    clearError();

    const review = reviewInput.value.trim();

    if (!review) {

        showError("Please enter a movie review.");

        return;
    }


    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";


    try {

        const response = await fetch(
            `${API_BASE_URL}/predict`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    text: review
                })
            }
        );


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail || "Prediction failed."
            );
        }


        showResult(
            data.sentiment,
            data.confidence
        );

    }

    catch (error) {

        console.error(error);

        showError(
            "Unable to connect to the sentiment API. " +
            "Make sure the FastAPI server is running."
        );

    }

    finally {

        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze Review";

    }

});


// =========================
// CSV Button
// =========================

csvButton.addEventListener("click", () => {

    csvFileInput.click();

});


// =========================
// CSV Upload
// =========================

csvFileInput.addEventListener("change", async () => {

    clearError();

    const file = csvFileInput.files[0];


    if (!file) {
        return;
    }


    if (!file.name.toLowerCase().endsWith(".csv")) {

        showError("Please select a CSV file.");

        csvFileInput.value = "";

        return;
    }


    csvButton.disabled = true;
    csvButton.textContent = "Processing...";


    try {

        const formData = new FormData();

        formData.append("file", file);


        const response = await fetch(
            `${API_BASE_URL}/predict/csv`,
            {
                method: "POST",
                body: formData
            }
        );


        if (!response.ok) {

            let message = "CSV prediction failed.";

            try {

                const data = await response.json();

                message = data.detail || message;

            }

            catch {
                // Response was not JSON.
            }

            throw new Error(message);
        }


        const blob = await response.blob();


        // Create a temporary download URL
        const downloadUrl = window.URL.createObjectURL(blob);


        // Create temporary download link
        const link = document.createElement("a");

        link.href = downloadUrl;
        link.download = "predictions.csv";

        document.body.appendChild(link);

        link.click();

        link.remove();


        // Release the temporary URL
        window.URL.revokeObjectURL(downloadUrl);

    }

    catch (error) {

        console.error(error);

        showError(
            error.message || "Unable to process CSV file."
        );

    }

    finally {

        csvButton.disabled = false;
        csvButton.textContent = "Upload CSV";

        // Allow selecting the same file again
        csvFileInput.value = "";

    }

});