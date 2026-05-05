# GraphRAG vs Flat RAG Comparison Lab

Dự án này triển khai và so sánh hai phương pháp RAG (Retrieval-Augmented Generation) phổ biến hiện nay: **Flat RAG** (dựa trên Vector Search) và **GraphRAG** (dựa trên Đồ thị Tri thức).

## 🚀 Tính năng nổi bật
- **Atomic Triple Extraction**: Trích xuất bộ ba (Subject, Predicate, Object) đơn tử để tránh nhiễu dữ liệu.
- **Bi-directional Traversal**: Duyệt đồ thị theo cả hai chiều (Successors & Predecessors) để tối ưu khả năng tìm kiếm multi-hop.
- **Schema Enforcement**: Ép LLM tuân thủ lược đồ thực thể/quan hệ định sẵn giúp đồ thị sạch và chính xác.
- **Multi-Provider Support**: Hỗ trợ OpenAI, Google Gemini và Local LLM (Ollama - Qwen2.5).

## 🛠️ Cài đặt

1. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Cấu hình môi trường**:
   Tạo file `.env` từ file mẫu và điền API Key:
   ```env
   OPENAI_API_KEY=your_key
   GOOGLE_API_KEY=your_key
   LLM_PROVIDER=google # hoặc openai, ollama
   ```

3. **Sử dụng Ollama (Nếu chạy Local)**:
   ```bash
   ollama pull qwen2.5:7b
   ```

## 📈 Cách chạy
Chạy script chính để xây dựng đồ thị, tạo cơ sở dữ liệu vector và thực hiện benchmark:
```bash
python graph_rag_system.py
```

## 📂 Cấu trúc thư mục
- `graph_rag_system.py`: Script xử lý chính (Indexing, Graph Build, RAG, Benchmark).
- `tech_corpus.txt`: Dữ liệu đầu vào về các công ty công nghệ.
- `report.md`: Báo cáo phân tích kết quả so sánh chi tiết.
- `graph_viz.png`: Biểu đồ trực quan hóa đồ thị tri thức.

## 📝 Kết quả
Hệ thống sẽ tạo ra file `benchmark_results.csv` chứa kết quả so sánh câu trả lời của 20 câu hỏi thử nghiệm giữa hai phương pháp.
