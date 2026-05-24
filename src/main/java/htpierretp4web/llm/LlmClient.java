package htpierretp4web.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.rag.query.transformer.QueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import jakarta.enterprise.context.Dependent;
// Ancien import : @Dependent
// import jakarta.enterprise.context.Dependent;

import java.io.Serializable;
import java.time.Duration;
import java.util.List;
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
                .modelName("gemini-2.5-flash-lite")
                .timeout(Duration.ofSeconds(30))
                .logRequestsAndResponses(true)
                .build();

        // Phase 1 : Ingestion des documents
        DocumentParser parser = new ApacheTikaDocumentParser();
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);

        // Chargement et ingestion de rag.pdf
        Document docRag = ClassPathDocumentLoader.loadDocument("rag.pdf", parser);
        List<TextSegment> segmentsRag = splitter.split(docRag);
        List<Embedding> embeddingsRag = embeddingModel.embedAll(segmentsRag).content();
        embeddingStore.addAll(embeddingsRag, segmentsRag);

        // Chargement et ingestion de 1_spring_solid_ioc.pdf
        Document docSpring = ClassPathDocumentLoader.loadDocument("1_spring_solid_ioc.pdf", parser);
        List<TextSegment> segmentsSpring = splitter.split(docSpring);
        List<Embedding> embeddingsSpring = embeddingModel.embedAll(segmentsSpring).content();
        embeddingStore.addAll(embeddingsSpring, segmentsSpring);

        // -- Phase 2 : Récupération et réponse --

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Test 3 - QueryTransformer pour reformuler les questions ambiguës
        QueryTransformer queryTransformer = new CompressingQueryTransformer(model);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryTransformer(queryTransformer)
                .contentRetriever(contentRetriever)
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(20);

        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
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