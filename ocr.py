import PyPDF2
import numpy as np
import pandas as pd
from io import BytesIO
from google.cloud import vision
from collections import Counter, defaultdict
from sklearn.cluster import DBSCAN, AgglomerativeClustering


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


def merge_2d(blocks, eps=0.05, min_samples=30, min_blocks=3, verbose=False):

    # Group similar paragraphs with DBSCAN algorithm

    points = []
    for block in blocks:
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                for v in word.bounding_box.normalized_vertices:
                    points.append((v.x, v.y))
                v1, v2, v3, v4 = word.bounding_box.normalized_vertices
                points.append(((v1.x + v2.x) / 2, (v1.y + v2.y) / 2))
                points.append(((v2.x + v3.x) / 2, (v2.y + v3.y) / 2))
                points.append(((v3.x + v4.x) / 2, (v3.y + v4.y) / 2))
                points.append(((v4.x + v1.x) / 2, (v4.y + v1.y) / 2))
                points.append(((v1.x + v3.x) / 2, (v1.y + v3.y) / 2))
                points.append(((v2.x + v4.x) / 2, (v2.y + v4.y) / 2))

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    if verbose:
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

    n = 0
    data = []
    for i, block in enumerate(blocks):
        for j, paragraph in enumerate(block.paragraphs):
            cluster_ids = [labels[n] for _ in range(10) for _ in paragraph.words]
            best_cluster_id = Counter(cluster_ids).most_common(1)[0][0]
            n += len(cluster_ids)
            if best_cluster_id != -1:
                data.append([i, j, best_cluster_id])
    df = pd.DataFrame(data, columns=["block", "paragraph", "cluster"])

    new_paragraphs = []
    merged_paragraphs = []
    unmerged_paragraphs = []

    for cluster_id, group in df.groupby("cluster"):
        cluster = {block_id: rows.paragraph for block_id, rows in group.groupby("block")}
        if len(cluster) < min_blocks:
            continue
        if verbose:
            print(cluster_id)
        new_paragraph = vision.Paragraph()
        for block_id, paragraph_ids in cluster.items():
            block = blocks[block_id]
            for paragraph_id in paragraph_ids:
                paragraph = block.paragraphs[paragraph_id]
                for word in paragraph.words:
                    # NOTE: hopefully paragraphs are sorted by y-axis
                    new_paragraph.words.append(word)
                merged_paragraphs.append(paragraph)
            if verbose:
                block_texts = []
                for paragraph_id in paragraph_ids:
                    paragraph_text = ""
                    paragraph = block.paragraphs[paragraph_id]
                    for word in paragraph.words:
                        paragraph_text += get_text(word)
                    block_texts.append(paragraph_text)
                print(block_texts)
        new_paragraphs.append(new_paragraph)

    for block in blocks:
        for paragraph in block.paragraphs:
            if paragraph not in merged_paragraphs:
                unmerged_paragraphs.append(paragraph)

    return new_paragraphs + unmerged_paragraphs


def merge_1d(paragraphs, distance_threshold=0.05, sort_x = False, delimeter=" | "):

    points = []
    for paragraph in paragraphs:
        ys = []
        for word in paragraph.words:
            for v in word.bounding_box.normalized_vertices:
                ys.append(v.y)
        points.append((0, np.mean(ys)))

    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold).fit_predict(points)

    blocks = defaultdict(list)
    for paragraph, cluster_id in zip(paragraphs, labels):
        blocks[cluster_id].append(paragraph)

    lines = []
    for cluster_id, paragraphs in blocks.items():
        ys = []
        texts = []
        for paragraph in paragraphs:
            xs = []
            text = ""
            for word in paragraph.words:
                for v in word.bounding_box.normalized_vertices:
                    xs.append(v.x)
                    ys.append(v.y)
                text += get_text(word)
            texts.append([np.min(xs), text.strip().replace("\n", " ")])
        if sort_x:
            texts = sorted(texts, key=lambda x: x[0])
        texts = [text for _, text in texts]
        lines.append([np.mean(ys), delimeter.join(texts)])

    lines = sorted(lines, key=lambda x: x[0])
    texts = [line for _, line in lines]
    return texts


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
