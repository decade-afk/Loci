/**
 * Loci Flutter 示例应用
 * 演示本地 AI 推理功能
 */

import 'package:flutter/material.dart';
import 'loci_ffi.dart';

void main() {
  runApp(const LociDemoApp());
}

class LociDemoApp extends StatelessWidget {
  const LociDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Loci AI Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const LociHomePage(),
    );
  }
}

class LociHomePage extends StatefulWidget {
  const LociHomePage({super.key});

  @override
  State<LociHomePage> createState() => _LociHomePageState();
}

class _LociHomePageState extends State<LociHomePage> {
  final LociEngine _engine = LociEngine();
  final TextEditingController _promptController = TextEditingController();

  bool _isInitialized = false;
  bool _isGenerating = false;
  String _output = '';
  String _statusMessage = '点击 "初始化引擎" 开始';

  @override
  void dispose() {
    _engine.dispose();
    _promptController.dispose();
    super.dispose();
  }

  Future<void> _initializeEngine() async {
    setState(() {
      _statusMessage = '正在初始化引擎...';
    });

    // TODO: 替换为实际的模型路径
    // Android: /sdcard/models/model.gguf
    // iOS: NSBundle mainBundle resource path
    const modelPath = '/path/to/model.gguf';

    final success = _engine.init(
      modelPath,
      nThreads: -1,    // 自动检测 CPU 核心数
      nGpuLayers: -1,  // 使用全部 GPU 层（如果支持）
    );

    setState(() {
      _isInitialized = success;
      _statusMessage = success
          ? '✅ 引擎初始化成功！'
          : '❌ 初始化失败，请检查模型路径';
    });
  }

  Future<void> _generateText() async {
    if (!_isInitialized) {
      _showSnackBar('请先初始化引擎');
      return;
    }

    if (_promptController.text.isEmpty) {
      _showSnackBar('请输入提示词');
      return;
    }

    setState(() {
      _isGenerating = true;
      _statusMessage = '正在生成...';
      _output = '';
    });

    // 在单独的 isolate 中运行推理（避免阻塞 UI）
    // 注意：FFI 调用必须在同一线程，这里简化处理
    final prompt = _promptController.text;
    final output = _engine.generate(prompt, maxTokens: 100);

    setState(() {
      _isGenerating = false;
      if (output != null) {
        _output = output;
        _statusMessage = '✅ 生成完成';
      } else {
        _statusMessage = '❌ 生成失败';
      }
    });
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Loci AI Demo'),
        elevation: 2,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // 状态指示器
            Card(
              color: _isInitialized ? Colors.green.shade50 : Colors.grey.shade100,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  children: [
                    Icon(
                      _isInitialized ? Icons.check_circle : Icons.info_outline,
                      color: _isInitialized ? Colors.green : Colors.grey,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        _statusMessage,
                        style: const TextStyle(fontSize: 16),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // 初始化按钮
            if (!_isInitialized)
              ElevatedButton.icon(
                onPressed: _initializeEngine,
                icon: const Icon(Icons.power_settings_new),
                label: const Text('初始化引擎'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.all(16),
                  textStyle: const TextStyle(fontSize: 16),
                ),
              ),

            // 输入区域
            if (_isInitialized) ...[
              TextField(
                controller: _promptController,
                decoration: const InputDecoration(
                  labelText: '输入提示词',
                  hintText: '例如：写一首关于春天的诗',
                  border: OutlineInputBorder(),
                ),
                maxLines: 3,
                enabled: !_isGenerating,
              ),

              const SizedBox(height: 16),

              // 生成按钮
              ElevatedButton.icon(
                onPressed: _isGenerating ? null : _generateText,
                icon: _isGenerating
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.auto_awesome),
                label: Text(_isGenerating ? '生成中...' : '生成文本'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.all(16),
                  textStyle: const TextStyle(fontSize: 16),
                ),
              ),

              const SizedBox(height: 16),

              // 输出区域
              Expanded(
                child: Card(
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.all(16),
                    child: Text(
                      _output.isEmpty ? '生成结果将显示在这里...' : _output,
                      style: TextStyle(
                        fontSize: 16,
                        color: _output.isEmpty ? Colors.grey : Colors.black,
                      ),
                    ),
                  ),
                ),
              ),
            ],

            // 性能提示
            const SizedBox(height: 16),
            const Text(
              '💡 提示：首次推理会加载模型，需要等待几秒钟',
              style: TextStyle(fontSize: 12, color: Colors.grey),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
