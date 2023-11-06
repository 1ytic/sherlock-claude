import PyPDF2
import numpy as np
from io import BytesIO
from google.cloud import vision
from sklearn.cluster import AgglomerativeClustering


def get_text(w):
    text = ""
    for s in w.symbols:
        text += s.text
        if s.property.detected_break.type == vision.TextAnnotation.DetectedBreak.BreakType.SPACE:
            text += " "
        elif s.property.detected_break.type == vision.TextAnnotation.DetectedBreak.BreakType.SURE_SPACE:
            text += " "
        elif s.property.detected_break.type == vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE:
            text += " "
        elif s.property.detected_break.type == vision.TextAnnotation.DetectedBreak.BreakType.HYPHEN:
            text += " "
        elif s.property.detected_break.type == vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK:
            text += " "
    return text

def merge_lines(blocks, scale_threshold=2, delimeter=" | "):

    paragraphs = [p for b in blocks for p in b.paragraphs]

    word_heights = []
    for paragraph in paragraphs:
        for word in paragraph.words:
            v1, v2, v3, v4 = word.bounding_box.normalized_vertices
            word_heights.append(np.abs(v1.y - v4.y))
            word_heights.append(np.abs(v2.y - v3.y))
    
    threshold = np.mean(word_heights) * scale_threshold

    points = []
    for paragraph in paragraphs:
        ys = [v.y for v in paragraph.bounding_box.normalized_vertices]
        points.append((0, np.mean(ys)))

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
    )

    labels = clustering.fit_predict(points)

    lines = []
    for cluster_id in range(clustering.n_clusters_):
        ys = []
        texts = []
        for label, paragraph in zip(labels, paragraphs):
            if label == cluster_id:
                ys.extend([v.y for v in paragraph.bounding_box.normalized_vertices])
                text = ""
                for word in paragraph.words:
                    text += get_text(word)
                texts.append(text.strip())
        lines.append([np.mean(ys), delimeter.join(texts)])

    lines = sorted(lines, key=lambda x: x[0])
    lines = [text for _, text in lines]
    return lines


def recognize_pdf_content(client, bytes):

    stream = BytesIO(bytes)

    pdf = PyPDF2.PdfReader(stream)
    num_pages = len(pdf.pages)

    input_config = vision.InputConfig(content=bytes, mime_type="application/pdf")

    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

    # context = vision.ImageContext(language_hints=["en-t-i0-handwrit"])

    page_nums = list(range(1, num_pages + 1))

    result = []

    for i in range(0, len(page_nums), 5):
        request = vision.AnnotateFileRequest(
            features=[feature],
            input_config=input_config,
            # image_context=context,
            pages=page_nums[i:i + 5],
        )
        # NOTE: Right now only one AnnotateFileRequest 
        # in BatchAnnotateFilesRequest is supported.
        batch_response: vision.BatchAnnotateFilesResponse = client.batch_annotate_files(
            requests=[request],
        )
        file_response: vision.AnnotateFileResponse = batch_response.responses[0]
        for image_response in file_response.responses:
            for page in image_response.full_text_annotation.pages:
                result.append(page)

    return result


def recognize_pdf_file(client, file_name):

    with open(file_name, "rb") as f:
        content = f.read()

    return recognize_pdf_content(client, content)
