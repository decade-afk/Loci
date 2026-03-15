package io.github.decadeafk.loci.sample

import android.database.Cursor
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import io.github.decadeafk.loci.sdk.LociDeviceSelector
import io.github.decadeafk.loci.sdk.LociEngine
import io.github.decadeafk.loci.sdk.LociException
import io.github.decadeafk.loci.sdk.LociRuntime
import java.io.File
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    private var engine: LociEngine? = null
    private var modelFile: File? = null

    private lateinit var selectedModelPath: TextView
    private lateinit var statusText: TextView
    private lateinit var outputText: TextView
    private lateinit var contextSizeInput: EditText
    private lateinit var maxTokensInput: EditText
    private lateinit var temperatureInput: EditText
    private lateinit var promptInput: EditText
    private lateinit var pickModelButton: Button
    private lateinit var loadModelButton: Button
    private lateinit var generateButton: Button
    private lateinit var streamButton: Button

    private val openModelFile =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) {
                importModel(uri)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        selectedModelPath = findViewById(R.id.selectedModelPath)
        statusText = findViewById(R.id.statusText)
        outputText = findViewById(R.id.outputText)
        contextSizeInput = findViewById(R.id.contextSizeInput)
        maxTokensInput = findViewById(R.id.maxTokensInput)
        temperatureInput = findViewById(R.id.temperatureInput)
        promptInput = findViewById(R.id.promptInput)
        pickModelButton = findViewById(R.id.pickModelButton)
        loadModelButton = findViewById(R.id.loadModelButton)
        generateButton = findViewById(R.id.generateButton)
        streamButton = findViewById(R.id.streamButton)

        pickModelButton.setOnClickListener {
            openModelFile.launch(arrayOf("*/*"))
        }

        loadModelButton.setOnClickListener {
            lifecycleScope.launch {
                loadEngine()
            }
        }

        generateButton.setOnClickListener {
            lifecycleScope.launch {
                runGeneration(stream = false)
            }
        }

        streamButton.setOnClickListener {
            lifecycleScope.launch {
                runGeneration(stream = true)
            }
        }

        lifecycleScope.launch {
            loadRuntimeInfo()
        }
    }

    override fun onDestroy() {
        engine?.close()
        engine = null
        super.onDestroy()
    }

    private fun importModel(uri: Uri) {
        setStatus("Importing model into app-private storage...")
        lifecycleScope.launch {
            try {
                val copiedFile = withContext(Dispatchers.IO) {
                    copyModelToPrivateStorage(uri)
                }
                modelFile = copiedFile
                selectedModelPath.text = copiedFile.absolutePath
                setStatus("Model is ready to load.")
            } catch (e: Exception) {
                setStatus("Model import failed: ${e.message}")
            }
        }
    }

    private suspend fun loadEngine() {
        val file = modelFile
        if (file == null) {
            setStatus("Pick a GGUF model file first.")
            return
        }

        val contextSize = contextSizeInput.text.toString().toIntOrNull() ?: 2048
        setBusy(true)
        setStatus("Loading model...")

        try {
            val loadedEngine = withContext(Dispatchers.IO) {
                LociEngine.createAuto(file.absolutePath, contextSize)
            }
            engine?.close()
            engine = loadedEngine
            setStatus("Engine loaded successfully.")
        } catch (e: Exception) {
            setStatus("Engine load failed: ${e.message}")
        } finally {
            setBusy(false)
        }
    }

    private suspend fun loadRuntimeInfo() {
        try {
            val summary = withContext(Dispatchers.IO) {
                val version = LociRuntime.version()
                val devices = LociDeviceSelector.create().use { selector ->
                    selector.listDevices()
                }
                val deviceSummary = devices.joinToString { "${it.name} (${it.deviceType.name})" }
                "Loci $version ready. Devices: $deviceSummary"
            }
            setStatus(summary)
        } catch (e: Exception) {
            setStatus("Loci runtime probe failed: ${e.message}")
        }
    }

    private suspend fun runGeneration(stream: Boolean) {
        val loadedEngine = engine
        if (loadedEngine == null) {
            setStatus("Load the model before generating.")
            return
        }

        val prompt = promptInput.text.toString().trim()
        if (prompt.isEmpty()) {
            setStatus("Enter a prompt first.")
            return
        }

        val maxTokens = maxTokensInput.text.toString().toIntOrNull() ?: 128
        val temperature = temperatureInput.text.toString().toFloatOrNull() ?: 0.7f

        outputText.text = ""
        setBusy(true)
        setStatus(if (stream) "Streaming generation..." else "Generating response...")

        try {
            if (stream) {
                withContext(Dispatchers.IO) {
                    loadedEngine.generateStream(
                        prompt = prompt,
                        maxTokens = maxTokens,
                        temperature = temperature,
                    ) { token ->
                        runOnUiThread {
                            outputText.append(token)
                        }
                        true
                    }
                }
            } else {
                val response = withContext(Dispatchers.IO) {
                    loadedEngine.generate(
                        prompt = prompt,
                        maxTokens = maxTokens,
                        temperature = temperature,
                    )
                }
                outputText.text = response
            }
            setStatus("Generation completed.")
        } catch (e: LociException) {
            setStatus("Generation failed: ${e.message}")
        } catch (e: Exception) {
            setStatus("Unexpected error: ${e.message}")
        } finally {
            setBusy(false)
        }
    }

    private suspend fun copyModelToPrivateStorage(uri: Uri): File {
        val modelsDir = File(filesDir, "models")
        if (!modelsDir.exists() && !modelsDir.mkdirs()) {
            throw IOException("Unable to create model directory")
        }

        val displayName = queryDisplayName(uri) ?: "model.gguf"
        val sanitizedName = displayName.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val destination = File(modelsDir, sanitizedName)

        contentResolver.openInputStream(uri)?.use { input ->
            destination.outputStream().use { output ->
                input.copyTo(output)
            }
        } ?: throw IOException("Unable to open selected model file")

        return destination
    }

    private fun queryDisplayName(uri: Uri): String? {
        val cursor: Cursor? = contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null,
        )

        cursor.use {
            if (it != null && it.moveToFirst()) {
                val index = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (index >= 0) {
                    return it.getString(index)
                }
            }
        }
        return null
    }

    private fun setBusy(busy: Boolean) {
        pickModelButton.isEnabled = !busy
        loadModelButton.isEnabled = !busy
        generateButton.isEnabled = !busy
        streamButton.isEnabled = !busy
    }

    private fun setStatus(message: String) {
        statusText.text = message
    }
}
