# Java REST Template

## Option A: Maven

```bash
mvn exec:java
```

## Option B: JDK only (no Maven)

```bash
javac -d out src/main/java/com/loci/LociRestClient.java
java -cp out com.loci.LociRestClient
```

Optional env var:

```bash
set LOCI_BASE_URL=http://127.0.0.1:8080
```
