const stepSections = [
    document.getElementById("step-1"),
    document.getElementById("step-2"),
    document.getElementById("step-3"),
];
const stepIndicators = document.querySelectorAll("[data-step-indicator]");
const form = document.getElementById("assessment-form");
const resultsSection = document.getElementById("results");
const predictionBanner = document.getElementById("prediction-banner");
const confidenceBar = document.getElementById("confidence-bar");
const confidencePercent = document.getElementById("confidence-percent");
const shapExplanation = document.getElementById("shap-explanation");
const modelComparison = document.getElementById("model-comparison");
const submitButton = document.getElementById("submit-button");

function showStep(stepNumber) {
    stepSections.forEach((section, index) => {
        section.hidden = index !== stepNumber - 1;
    });

    stepIndicators.forEach((indicator) => {
        indicator.classList.toggle(
            "active",
            Number(indicator.dataset.stepIndicator) === stepNumber
        );
    });
}

function validateStep(stepNumber) {
    const currentStep = document.getElementById(`step-${stepNumber}`);
    const fields = currentStep.querySelectorAll("input, select");

    for (const field of fields) {
        if (!field.checkValidity()) {
            field.reportValidity();
            return false;
        }
    }

    return true;
}

function encodeYesNo(value) {
    return value === "Yes" ? 1 : 0;
}

function encodeCompanySize(value) {
    const mapping = {
        "1-5": 0,
        "6-25": 1,
        "26-100": 2,
        "100-500": 3,
        "500-1000": 4,
        "More than 1000": 5,
    };
    return mapping[value] ?? 0;
}

function encodeWorkInterfere(value) {
    const mapping = {
        Never: 0,
        Rarely: 1,
        Sometimes: 2,
        Often: 3,
    };
    return mapping[value] ?? 0;
}

function encodeRemoteWork(value) {
    const mapping = {
        Yes: 1,
        No: 0,
        Sometimes: 0,
    };
    return mapping[value] ?? 0;
}

function encodeBenefits(value) {
    const mapping = {
        "Don't know": 0,
        No: 1,
        Yes: 2,
    };
    return mapping[value] ?? 0;
}

function encodeCareOptions(value) {
    const mapping = {
        No: 0,
        "Not sure": 1,
        Yes: 2,
    };
    return mapping[value] ?? 0;
}

function encodeSeekHelp(value) {
    const mapping = {
        "Don't know": 0,
        No: 1,
        Yes: 2,
    };
    return mapping[value] ?? 0;
}

function buildPayload() {
    const gender = document.getElementById("gender").value;

    return {
        Age: Number(document.getElementById("age").value),
        self_employed: encodeYesNo(document.getElementById("self-employed").value),
        family_history: encodeYesNo(document.getElementById("family-history").value),
        work_interfere: encodeWorkInterfere(document.getElementById("work-interfere").value),
        no_employees: encodeCompanySize(document.getElementById("company-size").value),
        remote_work: encodeRemoteWork(document.getElementById("remote-work").value),
        tech_company: encodeYesNo(document.getElementById("tech-company").value),
        benefits: encodeBenefits(document.getElementById("benefits").value),
        care_options: encodeCareOptions(document.getElementById("care-options").value),
        wellness_program: 0,
        seek_help: encodeSeekHelp(document.getElementById("seek-help").value),
        anonymity: 1,
        leave: 2,
        mental_health_consequence: 0,
        phys_health_consequence: 0,
        coworkers: 1,
        supervisor: 1,
        mental_health_interview: 0,
        phys_health_interview: 0,
        mental_vs_physical: 1,
        obs_consequence: 0,
        Gender_Female: gender === "Female" ? 1 : 0,
        Gender_Male: gender === "Male" ? 1 : 0,
        Gender_Other: gender === "Other" ? 1 : 0,
    };
}

function showErrorMessage() {
    let errorDiv = document.getElementById("error-message");

    if (!errorDiv) {
        errorDiv = document.createElement("div");
        errorDiv.id = "error-message";
        errorDiv.className = "error-message";
        form.insertAdjacentElement("afterend", errorDiv);
    }

    errorDiv.textContent = "Something went wrong. Please try again.";
}

function clearErrorMessage() {
    const errorDiv = document.getElementById("error-message");
    if (errorDiv) {
        errorDiv.remove();
    }
}

function renderShapExplanation(items) {
    const maxAbsValue = Math.max(
        ...items.map((item) => Math.abs(item.shap_value)),
        1
    );

    const shapListMarkup = items
        .map((item) => {
            const widthPercent = (Math.abs(item.shap_value) / maxAbsValue) * 100;
            return `
                <div class="shap-item">
                    <div class="shap-meta">
                        <span class="shap-label">${item.feature}</span>
                        <span class="shap-direction">${item.direction}</span>
                    </div>
                    <div class="shap-bar-track">
                        <div class="shap-bar" style="width: ${widthPercent}%;"></div>
                    </div>
                </div>
            `;
        })
        .join("");

    shapExplanation.innerHTML = `
        <h3>Top SHAP Drivers</h3>
        <div class="shap-list">${shapListMarkup}</div>
    `;
}

function renderResults(data) {
    resultsSection.hidden = false;
    resultsSection.classList.remove("negative");
    predictionBanner.classList.remove("positive", "negative");

    const isLikely = data.prediction.includes("Likely");
    predictionBanner.textContent = data.prediction;
    predictionBanner.classList.add(isLikely ? "positive" : "negative");
    if (!isLikely) {
        resultsSection.classList.add("negative");
    }

    confidenceBar.style.width = "0%";
    requestAnimationFrame(() => {
        confidenceBar.style.width = `${data.confidence_percent}%`;
    });
    confidencePercent.textContent = `${data.confidence_percent}%`;

    renderShapExplanation(data.shap_explanation || []);

    modelComparison.innerHTML = `
        <h3>Model Comparison</h3>
        <p>Random Forest: ${data.prediction} (${(data.probability * 100).toFixed(1)}%)</p>
        <p>Logistic Regression: ${data.lr_prediction} (${(data.lr_probability * 100).toFixed(1)}%)</p>
    `;

    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

document.getElementById("step-1-next").addEventListener("click", () => {
    if (validateStep(1)) {
        showStep(2);
    }
});

document.getElementById("step-2-next").addEventListener("click", () => {
    if (validateStep(2)) {
        showStep(3);
    }
});

document.getElementById("step-2-back").addEventListener("click", () => showStep(1));
document.getElementById("step-3-back").addEventListener("click", () => showStep(2));

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!validateStep(3)) {
        return;
    }

    clearErrorMessage();
    submitButton.innerHTML = "Analyzing...";
    submitButton.disabled = true;

    try {
        const response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(buildPayload()),
        });

        if (!response.ok) {
            throw new Error("Prediction request failed");
        }

        const data = await response.json();
        renderResults(data);
    } catch (error) {
        showErrorMessage();
    } finally {
        submitButton.innerHTML = "Submit";
        submitButton.disabled = false;
    }
});

showStep(1);
