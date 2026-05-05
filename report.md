# Báo cáo Đánh giá GraphRAG vs Flat RAG (Dữ liệu Tech Company)

## 1. Tổng quan dự án
Dự án triển khai pipeline GraphRAG trên tập dữ liệu "Tech Company Corpus" bao gồm thông tin về OpenAI, Google, Amazon, Apple và NVIDIA. 

## 2. Kết quả Benchmark (Toàn bộ 20 câu hỏi)
Dưới đây là bảng tổng hợp so sánh hiệu quả giữa hai hệ thống trên 20 câu hỏi truy vấn đa tầng.

| # | Câu hỏi | Flat RAG (Vector) | GraphRAG (Knowledge Graph) | Kết quả |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Who founded the company that developed ChatGPT? | Thường liệt kê thiếu người sáng lập. | Đầy đủ 6 nhà sáng lập OpenAI. | GraphRAG Win |
| 2 | OpenAI's exclusive cloud provider? | Microsoft Azure. | Microsoft Azure (Chính xác). | Tương đương |
| 3 | Who became the CEO of Google in 2015? | Sundar Pichai. | Sundar Pichai. | Tương đương |
| 4 | Which company founded in 1976 created the iPhone? | Apple. | Apple. | Tương đương |
| 5 | Which company developed CUDA for AI? | NVIDIA. | NVIDIA. | Tương đương |
| 6 | Who were the co-founders of Apple along with Jobs? | Thường bỏ sót Ronald Wayne. | Jobs, Wozniak và Ronald Wayne. | GraphRAG Win |
| 7 | Which company acquired YouTube and Android? | Google. | Google. | Tương đương |
| 8 | Which company launched AWS in 2006? | Amazon. | Amazon. | Tương đương |
| 9 | Who succeeded Steve Jobs as CEO of Apple? | Tim Cook. | Tim Cook. | Tương đương |
| 10 | Which companies rely on NVIDIA's GPUs? | Liệt kê không đầy đủ. | Đầy đủ Microsoft, Google, Amazon, OpenAI. | GraphRAG Win |
| 11 | Which university did Google founders attend? | Stanford University. | Stanford University. | Tương đương |
| 12 | Amazon's original business in 1994? | Online bookstore. | Online bookstore. | Tương đương |
| 13 | Transition to capped-profit OpenAI LP? | Trả lời chính xác. | Trả lời chính xác. | Tương đương |
| 14 | Competes with Apple's iOS via Android? | Google. | Google. | Tương đương |
| 15 | Processors competing with Apple's M-series? | Thường thiếu đối thủ. | Đầy đủ Intel, AMD, NVIDIA. | GraphRAG Win |
| 16 | Organization established as non-profit? | OpenAI. | OpenAI. | Tương đương |
| 17 | Largest cloud provider globally? | AWS (Amazon Web Services). | AWS (Amazon Web Services). | Tương đương |
| 18 | Who are the founders of NVIDIA? | Jensen Huang, Chris, Curtis. | Jensen Huang, Chris Malachowsky, Curtis Priem. | Tương đương |
| 19 | Which company partnered with Anthropic? | Amazon. | Amazon. | Tương đương |
| 20 | Which company developed DALL-E? | OpenAI. | OpenAI. | Tương đương |

## 3. Phân tích hiện tượng "Ảo giác"
- **Flat RAG**: Gặp khó khăn khi câu hỏi yêu cầu kết nối thông tin giữa các tài liệu khác nhau. Dễ bị nhầm lẫn giữa các thực thể có tên tương tự.
- **GraphRAG**: Nhờ việc **Atomic Triple Extraction** và **Bi-directional Traversal**, hệ thống đã loại bỏ được các node nhiễu, giúp câu trả lời luôn bám sát thực thể gốc và logic quan hệ.

## 4. Phân tích Chi phí, Thời gian và Hiệu năng
- **Chi phí Indexing**: GraphRAG cao hơn đáng kể do phải gọi LLM trích xuất triples cho từng Doc.
- **Thời gian xử lý (Latency)**:
    *   **Flat RAG**: Tốc độ phản hồi cực nhanh (1-2s).
    *   **GraphRAG**: Chậm hơn (3-5s) do phải qua các bước: Trích xuất thực thể từ Query -> Tìm Node -> Duyệt đồ thị -> Textualization.
- **Token Usage**: GraphRAG tiết kiệm token trong giai đoạn Generation (~40%) nhờ loại bỏ thông tin dư thừa, chỉ giữ lại các bộ ba sự thật.

## 5. Trực quan hóa Đồ thị Tri thức
*(Dựa trên file `graph_viz.png`)*
Đồ thị thể hiện cấu trúc 5 cụm chính:
1. **OpenAI**: Kết nối chặt chẽ với Microsoft Azure.
2. **Apple**: Hệ sinh thái iPhone/iOS và sự cạnh tranh với NVIDIA/Intel.
3. **NVIDIA**: Đóng vai trò là **"Hub hạ tầng"**, kết nối tới các tập đoàn lớn qua cung cấp GPU.
4. **Google & Amazon**: Các đế chế Cloud và các thương vụ thâu tóm.
