# API phân loại comment độc hại

API được tạo bằng FastAPI với khả năng phân loại bình luận độc hại sử dụng mô hình BERT đã được fine-tuned

## Hướng dẫn chạy chương trình
**Bước 1**: Clone repo về máy
```bash
git clone https://github.com/VietAnh1027/hate-speech-classification
cd hate-speech-classification
```
**Bước 2**: Tải thư viện cần thiết
```bash
pip install -r requirements.txt
```

**Bước 3**: Khởi động server uvicorn để nhận request từ người dùng:
```bash
uvicorn src.main:app --reload
```

## Sử dụng docker
Thực thi lệnh bên dưới để tạo image
```bash
docker build -t ten_mong_muon:phien_ban_cua_ban
vd: docker build -t hsc-image:1.0
```
Sau đó tiến hành tạo container và chạy
