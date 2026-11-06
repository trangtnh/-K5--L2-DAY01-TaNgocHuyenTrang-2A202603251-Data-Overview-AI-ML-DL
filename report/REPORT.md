# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** T4 GPU (CUDA)

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.11.0+cu130 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không thay đổi checkpoint, threshold hoặc mã nguồn thực hiện prediction. Có thêm 1 ô code đọc và trích xuất record kitchen từ các file JSON đã tạo, không làm thay đổi kết quả mô hình

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name` = cab ; `rank`=1 ; `score`= 0.510915; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record mô tả dự đoán cho toàn bộ ảnh traffic, không phải 1 vật thể riêng. Mô hình Classification xếp hạng các lớp cho cả ảnh mà không xác định vị trí hoặc ranh giới của từng đối tượng
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách lớp được xác định sẵn theo taxonomy ImageNet-1K gắn với checkpoint yolo11n-cls.pt, không phải do mô hình tự tạo ra trong quá trình dự đoán
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
class_id định danh lớp trong taxonomy, class_name thể hiện tên lớp để con người đọc hiểu, còn taxonomy_name xác định hệ thống phân loại đang sử dụng. Ghi đủ 3 trường giúp truy xuất và diễn giải nhãn chính xác, tránh nhầm lẫn giữa các taxonomy có thể có mã hoặc định nghĩa lớp khác nhau
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định tiêu chí xác định chủ thể chính của ảnh, như dựa vào mục tiêu bộ dữ liệu, mức độ nổi bật hoặc diện tích xuất hiện của đối tưởng. Đồng thời cần quy định cách xử lý khi nhiều lớp cùng phù hợp, không có lớp phù hợp hoặc khộgn thẻ xác định rõ chủ thể chính
6. Vì sao không được lấy `score` của model làm ground truth?
MOdel score chỉ phản ánh điểm mô hình gán cho 1 prediction, không xác nhận prediction đúng. Trong ảnh traffic, mô hình dự đoán top-1 là cab với score 0.510915, trong khi ảnh chứa nhiều loại phương tiện. Do đó cần đối chiếu ảnh và guideline để xác định nhãn ground truth, không sử dụng score thay thế quá trình gán nhãn

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person; `score`= 0.912625; `bbox_xyxy`= [385.33, 69.24, 498.92, 348.92]; `bbox_width`= 113.58; `bbox_height`= 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Bounding box bao quanh người đứng ở khu vực giữa bên phải ảnh kitchen. Từ góc trên-trái tại (385.33, 69.24), đến góc dưới-phải tại (498.92, 348.92), tính theo pixel với gốc toạ đọ (0, 0) ở góc trên bên trái ảnh
Box có chiều rộng 113.58 pixel và chiều cao 279.68 pixel. Đây là vùng hình chữ nhật mô hình dự đoán bao quanh person
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Threshold 0.20: 17 prediction
Threhold 0.60: 6 prediction
Ở ngưỡng mặc định 0.35 có 11 prediction. Khi tăng threshold từ 0.20 lên 0.60, số prediction giảm từ 17 -> 6 vì các prediction có score thấp hơn nne ngưỡng bị lọc bỏ 
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Cần kiểm tra thê các prediction được giữ lại, đặc biệt những prediction có score thấp. Cần đối chiếu ảnh gốc để phát hiện sai lớp, box lệch, trùng hoặc vùng không có vật thể tương ứng. Threshold chỉ lọc prediction, không quyết định vật nào phải có ground truth
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Mỗi vật thể thuộc phạm vi gắn nhãn phải được tạo 1 bounding box riêng, bao sát vùng vật thể theo quy định của guideline, hạn chế vùng nền không liên quan và không gộp nhiều vật thể độc lập vào cùng một box
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trong ảnh kitchen, một người chỉ xuất hiện một phần tại mép trái ảnh. Đây là trường hợp vật thể bị cắt bởi ranh giới ảnh. Mình sẽ viết thành quy tắc trong guideline quy định thống nhất điều kiện gán nhãn và cách đặt bounding box cho vật thể bị cắt mép. Nếu phần nhìn thấy không đủ xác định lớp hoặc trường hợp cụ thể chưa được guideline quy định, sẽ cần escalation lên Lab Coach thay vì tự quyết định

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-001 ; `class_name`= person; `score`= 0.899318; số điểm polygon = `polygon_point_count`= 348; 3–5 điểm đầu của `polygon_xy`= [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Bounding box chỉ xác định vùng hình chữ nhật bao quanh vật thể, còn polygon mô tả đường biên chi tiết bằng các điểm tọa độ, thể hiện hình dạng và vùng chiếm dụng của vật thể chính xác hơn so với hình chữ nhật, đặc biệt với đối tượng có hình dạng không đều.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để phân biệt từng đối tượng riêng biệt trong output bài lab, kể cả các đối tượng thuộc cùng một lớp
Nó không phải class ID vì class ID xác định loại đối tượng, còn instance ID xác định một đối tượng cụ thể. Nó cũng không phải tracking ID vì không dùng để duy trì danh tính đối tượng qua nhiều khung hình
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Với mỗi instance cần gán nhãn, annotator phải vẽ polygon bám sát đường biên nhìn thấy của vật thể, tránh lấy vùng nền hoặc gộp các vật thể độc lập. Phần bị che khuất hoặc ranh giới không rõ phải được xử lý thống nhất theo guideline
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trong ảnh kitchen, vùng dụng cụ treo bên phải có prediction spoon với score khoảng 0.49. Vùng mask dự đoán có ranh giới khó phân biệt giữa các dụng cụ nằm gần nhau, có nguy cơ gộp nhiều vật thể vào một instance. Reviewer cần đối chiếu ảnh gốc và guideline để kiểm tra class, đường biên và việc tách instance. Nếu guideline chưa quy định rõ cách xử lý những dụng cụ chồng lấn hoặc khó phân biệt, cần escalation lên Lab Coach

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn ảnh trong bài toán single-label | ảnh traffic có nhiều phương tiện nhưng Top-1 là cab(0.510915), chưa đại diện đầy đủ nội dung ảnh | Xác định lớp chính theo taxonomy và guidelinde | Kiểm tra tính phù hợp của lớp với ảnh và quy chắc chọn nhãn |
| Phát hiện vật thể | Một class và một bounding box cho mỗi vật thể thuộc phạm vi gắn nhãn | Ảnh kitchen có ngừoi bị cắt ở mép trái; cách xác định box cần thống nhất | Xác định từng vật thể, gán class và vẽ box theo guideline | Check sai lớp, box lệnh, trùng và vật thể bị bỏ sót |
| Instance segmentation | Một clas và một mask/polygon riêng cho môi9x instance | Vùng spoon(~0.49) trong ảnh kitchen có ranh giới khó phân biẹt giữa các dụng cụ | Gán class, tách instance và vẽ polygon theo biên vật thể | Kiểm tra độ bám biên, vùng nên bị lấy thừa, vật thể bị gộp hoặc bỏ sót |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng ba ảnh COCO công khai được notebook cung cấp; không tải ảnh cá nhân, dữ liệu khách hàng, dữ liệu nội bộ hoặc thôgn tin nhạy cảm lên Colab và Github công khai
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Dừng xử lý và không upload hoặc chia sẻ dữ liệu đó, đồng thừi báo Lab Coach/ giảng viên để được hướng dẫn xử lý theo quy định

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS