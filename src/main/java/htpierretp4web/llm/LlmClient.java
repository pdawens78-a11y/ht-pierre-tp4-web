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

/**
 * Gère l'interface avec l'API de Gemini.
 * Son rôle est essentiellement de lancer une requête à chaque nouvelle
 * question qu'on veut envoyer à l'API.
 *
 * De portée dependent pour réinitialiser la conversation à chaque fois que
 * l'instance qui l'utilise est renouvelée.
 * Par exemple, si l'instance qui l'utilise est de portée View, la conversation est
 * réunitialisée à chaque fois que l'utilisateur quitte la page en cours.
 */
@Dependent
public class LlmClient implements Serializable {

    // Rôle système choisi par l'utilisateur
    private String systemRole;

    // Service IA
    private Assistant assistant;

    // Mémoire de conversation
    private ChatMemory chatMemory;

    public LlmClient() {
        // Récupère la clé secrète pour travailler avec l'API du LLM, mise dans une variable d'environnement
        // du système d'exploitation.
        String key = System.getenv("GEMINI_KEY");

        if (key == null) {
            throw new RuntimeException("Clé GEMINI_KEY manquante !");
        }

        // Création du modèle Gemini
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(key)
                .modelName("gemini-2.5-flash")
                .timeout(Duration.ofSeconds(30))
                .build();

        // Mémoire de conversation
        this.chatMemory = MessageWindowChatMemory.withMaxMessages(20);

        // Création du service IA
        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .build();
    }

    /**
     * Définit le rôle système.
     */
    public void setSystemRole(String systemRole) {

        // Sauvegarde du rôle système
        this.systemRole = systemRole;

        if (systemRole == null || systemRole.isBlank()) {
            return;
        }
        // Nouveau contexte :
        // on vide la mémoire
        this.chatMemory.clear();

        // Ajout du rôle système à la mémoire
        this.chatMemory.add(
                SystemMessage.from(systemRole)
        );
    }

    /**
     * Envoie une question au LLM.
     */
    public String envoyerQuestion(String role, String question) {
        if (role != null) {
            setSystemRole(role);
        }
        return assistant.chat(question);
    }
}