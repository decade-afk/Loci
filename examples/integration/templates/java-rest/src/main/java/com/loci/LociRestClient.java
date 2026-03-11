package com.loci;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public final class LociRestClient {
    private static final HttpClient CLIENT = HttpClient.newHttpClient();

    public static void main(String[] args) throws Exception {
        String base = System.getenv().getOrDefault("LOCI_BASE_URL", "http://127.0.0.1:8080");

        String health = get(base + "/v1/health");
        System.out.println("health: " + health);

        String info = get(base + "/v1/info");
        System.out.println("info: " + info);

        String body = """
                {
                  "prompt":"Hello from Java template",
                  "max_tokens":64,
                  "temperature":0.7
                }
                """;

        String generated = postJson(base + "/v1/generate", body);
        System.out.println("response: " + generated);
    }

    private static String get(String url) throws IOException, InterruptedException {
        HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofSeconds(20))
                .GET()
                .build();
        HttpResponse<String> resp = CLIENT.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() != 200) {
            throw new IOException("GET failed: " + resp.statusCode() + " " + resp.body());
        }
        return resp.body();
    }

    private static String postJson(String url, String jsonBody) throws IOException, InterruptedException {
        HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();
        HttpResponse<String> resp = CLIENT.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() != 200) {
            throw new IOException("POST failed: " + resp.statusCode() + " " + resp.body());
        }
        return resp.body();
    }
}
