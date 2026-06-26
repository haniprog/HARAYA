function setClassificationStyle(classificationEl, label) {
    classificationEl.classList.remove(
        "classification-safe",
        "classification-potential",
        "classification-harassment"
    );

    if (label === "Safe Interaction") {
        classificationEl.classList.add("classification-safe");
    } else if (label === "Potential Harassment") {
        classificationEl.classList.add("classification-potential");
    } else {
        classificationEl.classList.add("classification-harassment");
    }
}

function showResultState(elements, hasResult) {
    elements.resultPlaceholder.classList.toggle("hidden", hasResult);
    elements.resultContent.classList.toggle("hidden", !hasResult);
}

export function setLoadingState(elements, isLoading) {
    elements.analyzeBtn.disabled = isLoading || elements.analyzeBtn.disabled;
    if (isLoading) {
        elements.resultPlaceholder.classList.add("hidden");
        elements.resultContent.classList.remove("hidden");
        elements.classificationEl.textContent = "Analyzing conversation...";
        elements.confidenceEl.textContent = "Confidence: --";
        elements.scoreEl.textContent = "Final Score: --";
        elements.recommendationListEl.innerHTML = "";
        elements.reasonListEl.innerHTML = "";
        elements.legalBasisListEl.innerHTML = "";
        elements.featureBreakdownListEl.innerHTML = "";
    }
}

export function resetOutput(elements) {
    elements.classificationEl.textContent = "No analysis yet.";
    elements.confidenceEl.textContent = "Confidence: --";
    elements.scoreEl.textContent = "Final Score: --";
    elements.recommendationListEl.innerHTML = "";
    elements.legalBasisListEl.innerHTML = "";
    elements.featureBreakdownListEl.innerHTML = "";
    elements.escalationAnalysisListEl.innerHTML = "";

    elements.classificationEl.classList.remove(
        "classification-safe",
        "classification-potential",
        "classification-harassment"
    );

    showResultState(elements, false);
}

function formatFeatureLabel(key) {
    const labels = {
        bertScore: "BERT Score",
        frequency: "Frequency",
        repetition: "Repetition",
        sentiment: "Sentiment",
        sentimentTrend: "Sentiment Trend",
        keywordSignal: "Keyword Signal",
        escalationScore: "Escalation Score",
        supportFloor: "Support Floor"
    };

    return labels[key] || key;
}

function getEscalationLevel(result) {
    if (result.finalScore >= 0.75) return "Critical";
    if (result.finalScore >= 0.55) return "High";
    if (result.finalScore >= 0.35) return "Moderate";
    return "Low";
}

function getConversationProgression(result) {
    const trend = result.features?.sentimentTrend ?? 0;
    const frequency = result.features?.frequency ?? 0;

    if (result.finalScore >= 0.7 || trend < -0.15 || frequency >= 4) {
        return {
            label: "Harassment",
            stages: ["Safe", "Potential", "Harassment"],
            activeStage: 3
        };
    }

    if (result.finalScore >= 0.4 || trend < 0) {
        return {
            label: "Potential Harassment",
            stages: ["Safe", "Potential"],
            activeStage: 2
        };
    }

    return {
        label: "Safe Interaction",
        stages: ["Safe"],
        activeStage: 1
    };
}

export function renderResult(elements, result) {
    elements.classificationEl.textContent = `Result: ${result.label}`;
    elements.confidenceEl.textContent = `Confidence: ${result.confidence.toFixed(2)}`;
    elements.scoreEl.textContent = `Final Score: ${result.finalScore.toFixed(3)}`;

    setClassificationStyle(elements.classificationEl, result.label);
    showResultState(elements, true);

    elements.recommendationListEl.innerHTML = "";
    result.recommendations.forEach((action) => {
        const item = document.createElement("li");
        item.textContent = action;
        elements.recommendationListEl.appendChild(item);
    });

    elements.legalBasisListEl.innerHTML = "";
    result.legalBasis.forEach((basis) => {
        const item = document.createElement("li");
        item.textContent = basis;
        elements.legalBasisListEl.appendChild(item);
    });

    elements.featureBreakdownListEl.innerHTML = "";
    elements.featureBreakdownListEl.className = "feature-grid";

    const entries = Object.entries(result.features || {});

    entries.forEach(([key, value]) => {
        if (key === "keywordHits") return;

        const item = document.createElement("li");
        item.className = "feature-card";

        const label = document.createElement("span");
        label.className = "feature-label";
        label.textContent = formatFeatureLabel(key);

        const valueEl = document.createElement("strong");
        valueEl.className = "feature-value";

        if (typeof value === "number") {
            valueEl.textContent = value.toFixed(3);
        } else {
            valueEl.textContent = value;
        }

        item.appendChild(label);
        item.appendChild(valueEl);
        elements.featureBreakdownListEl.appendChild(item);
    });

    elements.escalationAnalysisListEl.innerHTML = "";
    elements.escalationAnalysisListEl.className = "feature-grid";

    const escalationItem = document.createElement("li");
    escalationItem.className = "feature-card";

    const escalationLabel = document.createElement("span");
    escalationLabel.className = "feature-label";
    escalationLabel.textContent = "Escalation Level";

    const escalationValue = document.createElement("strong");
    escalationValue.className = "feature-value";
    escalationValue.textContent = getEscalationLevel(result);

    escalationItem.appendChild(escalationLabel);
    escalationItem.appendChild(escalationValue);
    elements.escalationAnalysisListEl.appendChild(escalationItem);

    const progressionItem = document.createElement("li");
    progressionItem.className = "feature-card";

    const progressionLabel = document.createElement("span");
    progressionLabel.className = "feature-label";
    progressionLabel.textContent = "Conversation Progression";

    const progressionValue = document.createElement("div");
    progressionValue.className = "progression-path";

    const progression = getConversationProgression(result);
    const stages = progression.stages;

    stages.forEach((stage, index) => {
        const dot = document.createElement("span");
        dot.className = "progression-dot";
        if (index + 1 <= progression.activeStage) {
            dot.classList.add("is-active");
        }

        const label = document.createElement("span");
        label.className = "progression-stage";
        label.textContent = stage;

        const node = document.createElement("span");
        node.className = "progression-node";
        node.appendChild(dot);
        node.appendChild(label);

        progressionValue.appendChild(node);

        if (index < stages.length - 1) {
            const arrow = document.createElement("span");
            arrow.className = "progression-arrow";
            arrow.textContent = "→";
            progressionValue.appendChild(arrow);
        }
    });

    const progressionSummary = document.createElement("strong");
    progressionSummary.className = "feature-value";
    progressionSummary.textContent = progression.label;

    progressionItem.appendChild(progressionLabel);
    progressionItem.appendChild(progressionValue);
    progressionItem.appendChild(progressionSummary);
    elements.escalationAnalysisListEl.appendChild(progressionItem);
}
