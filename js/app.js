import { analyzeConversation } from "./core/api.js";
import { createMessageStore, renderConversation, renderConversationLog } from "./ui/messages.js";
import { renderResult, resetOutput, setLoadingState } from "./ui/render.js";

document.addEventListener("DOMContentLoaded", () => {
    //User Input Layer (dito collection ng mga inputs)
    const elements = {
        newAnalysisBtn: document.getElementById("newAnalysisBtn"),
        analyzeBtn: document.getElementById("analyzeBtn"),
        clearBtn: document.getElementById("clearConversationBtn"),
        composeForm: document.getElementById("composeForm"),
        senderInput: document.getElementById("senderInput"),
        senderBtnUser: document.getElementById("senderBtnUser"),
        senderBtnOther: document.getElementById("senderBtnOther"),
        messageInput: document.getElementById("messageInput"),
        conversationList: document.getElementById("conversationList"),
        conversationLog: document.getElementById("conversationLog"),
        resultPlaceholder: document.getElementById("resultPlaceholder"),
        resultContent: document.getElementById("resultContent"),
        classificationEl: document.getElementById("classification"),
        confidenceEl: document.getElementById("confidence"),
        scoreEl: document.getElementById("score"),
        recommendationListEl: document.getElementById("recommendationList"),
        reasonListEl: document.getElementById("escalationAnalysisList"),
        legalBasisListEl: document.getElementById("legalBasisList"),
        featureBreakdownListEl: document.getElementById("featureBreakdownList"),
        escalationAnalysisListEl: document.getElementById("escalationAnalysisList"),
        resultCard: document.querySelector(".result-card"),
        recommendationsPanel: document.getElementById("recommendationsPanel"),
        legalBasisPanel: document.getElementById("legalBasisPanel")
    };

    const messageStore = createMessageStore([]);
    let analysisResults = [];

    const setSender = (sender) => {
        elements.senderInput.value = sender;
        elements.senderBtnUser.classList.toggle("is-active", sender === "user");
        elements.senderBtnOther.classList.toggle("is-active", sender === "other");
        elements.messageInput.placeholder = sender === "user"
            ? "Type a message as User A..."
            : "Type a message as User B...";
    };

    const updateAnalyzeState = () => {
        elements.analyzeBtn.disabled = messageStore.isEmpty();
    };

    const setDecisionTheme = (label) => {
        elements.resultCard.classList.remove("decision-safe", "decision-potential", "decision-harassment");
        elements.recommendationsPanel.classList.remove("panel-safe", "panel-potential", "panel-harassment");
        elements.legalBasisPanel.classList.remove("panel-safe", "panel-potential", "panel-harassment");

        if (label === "Safe Interaction") {
            elements.resultCard.classList.add("decision-safe");
            elements.recommendationsPanel.classList.add("panel-safe");
            elements.legalBasisPanel.classList.add("panel-safe");
        } else if (label === "Potential Harassment") {
            elements.resultCard.classList.add("decision-potential");
            elements.recommendationsPanel.classList.add("panel-potential");
            elements.legalBasisPanel.classList.add("panel-potential");
        } else {
            elements.resultCard.classList.add("decision-harassment");
            elements.recommendationsPanel.classList.add("panel-harassment");
            elements.legalBasisPanel.classList.add("panel-harassment");
        }
    };

    const resetDecisionTheme = () => {
        elements.resultCard.classList.remove("decision-safe", "decision-potential", "decision-harassment");
        elements.recommendationsPanel.classList.remove("panel-safe", "panel-potential", "panel-harassment");
        elements.legalBasisPanel.classList.remove("panel-safe", "panel-potential", "panel-harassment");
    };

    messageStore.subscribe((messages) => {
        renderConversation(elements.conversationList, messages, analysisResults);
        renderConversationLog(elements.conversationLog, messages);
        updateAnalyzeState();
    });

    //Module 1: Input
    const addMessage = () => {
        analysisResults = [];

        const didAdd = messageStore.add({
            text: elements.messageInput.value,
            sender: elements.senderInput.value
        });
        if (!didAdd) return;

        resetOutput(elements);
        resetDecisionTheme();
        elements.messageInput.value = "";
    };

    const runAnalysis = async () => {
        if (messageStore.isEmpty()) {
            elements.messageInput.focus();
            return;
        }

        const conversationText = messageStore.getAll().map((entry) => entry.text);
        elements.analyzeBtn.disabled = true;
        setLoadingState(elements, true);

        try {
            const result = await analyzeConversation(conversationText);
            renderResult(elements, result);
        } catch (error) {
            console.error("HARAYA API unavailable.", error);
            alert("HARAYA backend is unavailable. Please start the Python API and try again.");
            resetOutput(elements);
        } finally {
            elements.analyzeBtn.disabled = messageStore.isEmpty();
            setLoadingState(elements, false);
        }
    };

    elements.composeForm.addEventListener("submit", (event) => {
        event.preventDefault();
        addMessage();
    });

    elements.analyzeBtn.addEventListener("click", runAnalysis);

    elements.senderBtnUser.addEventListener("click", () => setSender("user"));
    elements.senderBtnOther.addEventListener("click", () => setSender("other"));

    elements.clearBtn.addEventListener("click", () => {
        analysisResults = [];
        messageStore.clear();
        resetOutput(elements);
        resetDecisionTheme();
        setSender("user");
        elements.messageInput.focus();
    });

    elements.newAnalysisBtn.addEventListener("click", () => {
        analysisResults = [];
        messageStore.clear();
        resetOutput(elements);
        resetDecisionTheme();
        setSender("user");
        elements.messageInput.focus();
        window.scrollTo({ top: 0, behavior: "smooth" });
    });

    setSender("user");
    updateAnalyzeState();
    resetOutput(elements);
    resetDecisionTheme();
});
