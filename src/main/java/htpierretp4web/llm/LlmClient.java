package htpierretp4web.llm;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;

import jakarta.enterprise.context.Dependent;

import java.io.Serializable;
import java.time.Duration;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

@Dependent
public class LlmClient implements Serializable {

    private String systemRole;
    private Assistant assistant;
    private ChatMemory chatMemory;

    public LlmClient() {

        // Configuration du logging
        configureLogger();

        String key = System.getenv("GEMINI_KEY");
        if (key == null) {
            throw new RuntimeException("Clé GEMINI_KEY manquante !");
        }

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(key)
                .modelName("gemini-2.0-flash")
                .timeout(Duration.ofSeconds(30))
                .logRequestsAndResponses(true)
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(20);

        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .build();
    }

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public void setSystemRole(String systemRole) {
        this.systemRole = systemRole;
        if (systemRole == null || systemRole.isBlank()) {
            return;
        }
        this.chatMemory.clear();
        this.chatMemory.add(SystemMessage.from(systemRole));
    }

    public String envoyerQuestion(String role, String question) {
        if (role != null) {
            setSystemRole(role);
        }
        return assistant.chat(question);
    }
}