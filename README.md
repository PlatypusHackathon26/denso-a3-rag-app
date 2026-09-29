# Denso A3 Factory Chat

Ứng dụng gồm giao diện Node.js và dịch vụ RAG nội bộ dùng LightRAG. Node.js xác thực người dùng, lọc tài liệu theo quyền rồi gửi câu hỏi và tài liệu được phép sang Python. Python lập chỉ mục và truy vấn bằng Ollama.

## Chạy ứng dụng

1. Cài các gói Python: `pip install -r requirements.txt`.
2. Khởi động Ollama và tải model chat cùng model embedding:
   `ollama pull gemma4:26b`
   `ollama pull nomic-embed-text`
3. Chạy dịch vụ RAG: `python app.py`.
4. Trong terminal khác chạy `npm start`, rồi mở `http://localhost:3000`.

Mặc định, LightRAG gọi Ollama tại địa chỉ mặc định của thư viện. Có thể cấu hình `LIGHTRAG_LLM_MODEL`, `LIGHTRAG_EMBEDDING_MODEL`, `LIGHTRAG_WORKING_DIR` và `RAG_JSON_PATH`. `RAG_JSON_PATH` mặc định là `data.json` cạnh `app.py`. Có thể đặt cùng `RAG_API_TOKEN` cho tiến trình Node và Python để xác thực các lời gọi nội bộ.

## Nạp JSON và hỏi đáp

Khi có câu hỏi, `app.py` đọc file JSON và đưa nội dung được phép vào corpus của người dùng cùng các tài liệu SQLite được phép xem. Lần hỏi đầu tiên với một corpus sẽ lập chỉ mục; các lần sau dùng lại kho đã lưu dưới `lightrag_local_storage/authorized/`. Hỗ trợ một object, một mảng object hoặc `{ "documents": [...] }`. Mỗi object có thể có `title`, `doc_id`, `metadata`, `cleaned_text` hoặc `content`, `tables` và `images` như trong `data.json`.

LightRAG tự chia văn bản thành chunk (mặc định 1200 token, chồng lấn 100 token), gọi `nomic-embed-text` để tạo vector embedding, đồng thời dùng model chat để trích xuất thực thể và quan hệ làm chỉ mục đồ thị. Ollama dùng `gemma4:26b` để trả lời dựa trên kết quả truy xuất. Có thể đổi model và chunk size bằng `LIGHTRAG_LLM_MODEL`, `LIGHTRAG_EMBEDDING_MODEL`, `LIGHTRAG_CHUNK_TOKEN_SIZE` và `LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE`. `access_level` 1, 2, 3 lần lượt giới hạn tài liệu cho nhân viên, quản lý, giám đốc; nếu thiếu thì mặc định là nhân viên. Có thể dùng trực tiếp các tên vai trò `employee`, `manager`, `director`.

Không cần chạy `python setup_rag.py` cho luồng chat: script đó chỉ nạp JSON vào kho LightRAG độc lập, còn `app.py` xây corpus riêng theo quyền truy cập để không trộn dữ liệu của các vai trò. Script vẫn có thể dùng để nạp kho mặc định cho mục đích độc lập.

Tài liệu PDF tải lên từ giao diện tiếp tục được trích xuất và lưu trong SQLite. Luồng JSON nạp trực tiếp dữ liệu đã có; không cần làm sạch hoặc trích xuất lại PDF.
