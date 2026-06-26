const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

function getApiBaseUrl() {
    return window.HARAYA_API_BASE_URL || DEFAULT_API_BASE_URL;
}

function normalizeBackendResult(payload) {
    const probabilities = payload?.features?.probabilities || {};
    const behavioral = payload?.features?.behavioral || {};

    return {
        conversation: payload?.conversation || "",
        label: payload?.label || "Safe Interaction",
        finalScore: Number(payload?.finalScore || 0),
        confidence: Number(payload?.confidence || 0),
        features: {
            bertScore: Number(probabilities["Harassment"] || payload?.finalScore || 0),
            frequency: Number(behavioral.frequency || 0),
            repetition: Number(behavioral.repetition || 0),
            sentiment: Number(behavioral.sentiment || 0),
            keywordSignal: 0,
            escalationScore: Number(payload?.finalScore || 0),
            supportFloor: 0,
            keywordHits: {
                safe: 0,
                potential: 0,
                harassment: 0
            }
        },
        reasons: Array.isArray(payload?.reasons) ? payload.reasons : [],
        legalBasis: Array.isArray(payload?.legalBasis) ? payload.legalBasis : [],
        recommendations: Array.isArray(payload?.recommendations) ? payload.recommendations : []
    };
}

export async function analyzeConversation(messages) {
    const response = await fetch(`${getApiBaseUrl()}/analyze`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ messages })
    });

    if (!response.ok) {
        throw new Error(`HARAYA API request failed with status ${response.status}`);
    }

    const payload = await response.json();
    return normalizeBackendResult(payload);
}
