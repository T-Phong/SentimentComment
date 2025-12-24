from flask import Flask, request, jsonify
from model import predict_sentiment_3sentiment, predict_sentiment_5sentiment
import pandas as pd

# Khởi tạo Flask app
app = Flask(__name__)

# Xử lý encoding cho tiếng Việt để hiển thị đúng trong response
app.config['JSON_AS_ASCII'] = False

@app.route("/predict", methods=['POST'])
def predict():
    """
    Dự đoán cảm xúc từ văn bản đầu vào.
    Request body phải là JSON có dạng: {"text": "nội dung bình luận"}
    """
    # Lấy dữ liệu JSON từ request
    json_data = request.get_json()

    # Kiểm tra xem key 'text' có tồn tại và không rỗng không
    if not json_data or 'text' not in json_data or not json_data.get('text', '').strip():
        return jsonify({"error": "Vui lòng cung cấp trường 'text' trong request body."}), 400
    # Kiểm tra xem key 'type' có tồn tại và không rỗng không
    if not json_data or 'type' not in json_data or not json_data.get('type', '').strip():
        return jsonify({"error": "Vui lòng cung cấp trường 'type' trong request body."}), 400
    
    # Lấy văn bản từ dữ liệu
    text_to_predict = json_data['text']
    sentiment_type = json_data['type']
    # Gọi hàm dự đoán từ model
    if sentiment_type == "3sentiment":
        sentiment, score = predict_sentiment_3sentiment(text_to_predict)
    elif sentiment_type == "5sentiment":
        sentiment, score = predict_sentiment_5sentiment(text_to_predict)

    # Tạo response
    response = {
        "comment": text_to_predict,
        "sentiment": sentiment,
        "confidence": score
    }

    return jsonify(response)

@app.route("/")
def read_root():
    return jsonify({"message": "Chào mừng đến với API Phân tích Cảm xúc sử dụng Flask!"})

@app.route("/predict-batch", methods=['POST'])
def predict_batch():
    """
    Dự đoán cảm xúc cho một loạt bình luận từ file Excel.
    File Excel phải được gửi dưới dạng form-data với key là 'file'.
    Cột đầu tiên của file Excel sẽ được sử dụng làm cột chứa bình luận.
    """
    # 1. Kiểm tra xem có file trong request không
    if 'file' not in request.files:
        return jsonify({"error": "Không tìm thấy file trong request (key phải là 'file')."}), 400

    # Dữ liệu form đi kèm với file sẽ nằm trong request.form
    sentiment_type = request.form.get('type')

    # Kiểm tra xem key 'type' có tồn tại và không rỗng không
    if not sentiment_type or sentiment_type.strip() not in ["3sentiment", "5sentiment"]:
        return jsonify({"error": "Vui lòng cung cấp trường 'type' (3sentiment hoặc 5sentiment) trong form data."}), 400
    file = request.files['file']

    # 2. Kiểm tra xem người dùng có chọn file không
    if file.filename == '':
        return jsonify({"error": "Chưa chọn file nào."}), 400

    # 3. Kiểm tra định dạng file
    if not file.filename.endswith(('.xlsx', '.xls')):
        return jsonify({"error": "Định dạng file không hợp lệ. Vui lòng sử dụng file .xlsx hoặc .xls."}), 400

    try:
        # 4. Đọc file Excel bằng pandas, sử dụng engine openpyxl
        df = pd.read_excel(file, engine='openpyxl')

        if df.empty:
            return jsonify({"error": "File Excel rỗng."}), 400

        # Lấy tên cột đầu tiên để xử lý
        comments_column = df.columns[0]
        
        results = []
        # 5. Lặp qua từng bình luận (bỏ qua các dòng rỗng) để dự đoán
        for comment in df[comments_column].dropna().astype(str):
            if sentiment_type == "3sentiment":
                sentiment, score = predict_sentiment_3sentiment(comment)
            elif sentiment_type == "5sentiment":
                sentiment, score = predict_sentiment_5sentiment(comment)
            results.append({"comment": comment, "sentiment": sentiment, "confidence": score})
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Đã xảy ra lỗi khi xử lý file: {str(e)}"}), 500

if __name__ == "__main__":
    # Chạy app ở chế độ debug để tự động reload khi có thay đổi
    # host='0.0.0.0' để có thể truy cập từ bên ngoài network
    # Lưu ý: Không sử dụng debug=True trong môi trường production
    app.run(host="0.0.0.0", port=8000, debug=True)