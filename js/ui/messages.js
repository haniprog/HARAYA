function normalizeMessageEntry(entry) {
    if (typeof entry === "string") {
        return {
            text: entry,
            sender: "other"
        };
    }

    return {
        text: (entry.text || "").trim(),
        sender: entry.sender === "user" ? "user" : "other"
    };
}

export function createMessageStore(initialMessages = []) {
    const messages = initialMessages
        .map(normalizeMessageEntry)
        .filter((entry) => entry.text);
    const listeners = new Set();

    const notify = () => {
        const snapshot = [...messages];
        listeners.forEach((listener) => listener(snapshot));
    };

    return {
        getAll() {
            return [...messages];
        },
        isEmpty() {
            return messages.length === 0;
        },
        add({ text, sender }) {
            const value = (text || "").trim();
            if (!value) return false;

            messages.push({
                text: value,
                sender: sender === "user" ? "user" : "other"
            });
            notify();
            return true;
        },
        clear() {
            messages.length = 0;
            notify();
        },
        subscribe(listener) {
            listeners.add(listener);
            listener([...messages]);
            return () => listeners.delete(listener);
        }
    };
}

function getDisplayMix(result) {
    const score = result.finalScore;

    if (result.label === "Safe Interaction") {
        const safe = 0.79 + ((1 - score) * 0.08);
        const potential = 0.11 + (score * 0.03);
        const harassment = Math.max(0.05, 1 - safe - potential);
        return [safe, potential, harassment];
    }

    if (result.label === "Potential Harassment") {
        const safe = 0.35 + ((0.7 - score) * 0.05);
        const potential = 0.54 + ((score - 0.4) * 0.06);
        const harassment = Math.max(0.05, 1 - safe - potential);
        return [safe, potential, harassment];
    }

    const harassment = 0.68 + (score * 0.1);
    const potential = 0.2 - (score * 0.03);
    const safe = Math.max(0.05, 1 - harassment - potential);
    return [safe, potential, harassment];
}

function createBarRow(label, percent, toneClass) {
    const row = document.createElement("div");
    row.className = "analysis-row";

    const textLabel = document.createElement("span");
    textLabel.className = "analysis-row-label";
    textLabel.textContent = label;

    const track = document.createElement("div");
    track.className = "analysis-track";

    const fill = document.createElement("div");
    fill.className = `analysis-fill ${toneClass}`;
    fill.style.width = `${Math.max(4, Math.min(100, percent * 100))}%`;
    track.appendChild(fill);

    const value = document.createElement("span");
    value.className = "analysis-row-value";
    value.textContent = `${(percent * 100).toFixed(1)}%`;

    row.appendChild(textLabel);
    row.appendChild(track);
    row.appendChild(value);
    return row;
}

export function renderConversation(container, messages, analyses = []) {
    container.innerHTML = "";

    if (!messages.length) {
        const emptyState = document.createElement("p");
        emptyState.className = "message-empty";
        emptyState.textContent = "No conversation yet";
        container.appendChild(emptyState);
        return;
    }

    messages.forEach((message, index) => {
        const result = analyses[index];
        const hasAnalysis = Boolean(result && typeof result.label === "string");
        const toneClass = hasAnalysis
            ? (result.label === "Safe Interaction"
                ? "safe"
                : result.label === "Potential Harassment"
                    ? "potential"
                    : "harassment")
            : "";

        const entry = document.createElement("article");
        const bubble = document.createElement("article");
        const sideClass = message.sender === "user" ? "right" : "left";
        entry.className = `message-entry ${sideClass}`;
        bubble.className = `message-bubble ${sideClass}`;

        const avatar = document.createElement("span");
        avatar.className = `message-avatar ${sideClass}`;
        avatar.setAttribute("aria-hidden", "true");

        const title = document.createElement("h4");
        title.className = "message-title";
        title.textContent = message.text;

        bubble.appendChild(title);

        if (hasAnalysis) {
            const chip = document.createElement("span");
            chip.className = `message-chip ${toneClass}`;
            chip.textContent = `BERT | ${result.label} | ${result.confidence.toFixed(2)}`;
            bubble.appendChild(chip);
        }

        if (sideClass === "right") {
            entry.appendChild(bubble);
            entry.appendChild(avatar);
        } else {
            entry.appendChild(avatar);
            entry.appendChild(bubble);
        }
        container.appendChild(entry);
    });

    container.scrollTop = container.scrollHeight;
}

export function renderConversationLog(container, messages) {
    container.innerHTML = "";

    if (!messages.length) {
        const emptyState = document.createElement("p");
        emptyState.className = "conversation-log-empty";
        emptyState.textContent = "No messages yet.";
        container.appendChild(emptyState);
        return;
    }

    messages.forEach((message) => {
        const row = document.createElement("p");
        row.className = "conversation-log-row";
        row.textContent = `${message.sender === "user" ? "User_A" : "User_B"}: ${message.text}`;
        container.appendChild(row);
    });

    container.scrollTop = container.scrollHeight;
}
