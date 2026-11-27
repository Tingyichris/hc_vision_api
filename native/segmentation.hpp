#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

inline std::string formatPredictions(std::vector<float>& detections,
    std::vector<float>& prototypes, cv::Rect content, cv::Size inputSize,
    cv::Size sourceSize, int candidates, int maskChannels, int maskSize) {
  std::vector<cv::Rect2d> boxes;
  std::vector<float> scores;
  std::vector<int> anchors, selected;
  for (int anchor = 0; anchor < candidates; ++anchor) {
    const float score = detections[4 * candidates + anchor];
    if (!(score > 0.35f) || !std::isfinite(score)) continue;
    const float width = detections[2 * candidates + anchor];
    const float height = detections[3 * candidates + anchor];
    const float left = detections[anchor] - width / 2;
    const float top = detections[candidates + anchor] - height / 2;
    if (!(width > 0 && height > 0) || !std::isfinite(left + top + width + height)) continue;
    boxes.emplace_back(left, top, width, height);
    scores.push_back(score);
    anchors.push_back(anchor);
  }
  cv::dnn::NMSBoxes(boxes, scores, 0.35f, 0.50f, selected);
  std::ostringstream json;
  json << R"({"predictions":[)";
  bool first = true;
  const double scaleX = double(sourceSize.width) / content.width;
  const double scaleY = double(sourceSize.height) / content.height;
  const cv::Mat prototypeMatrix(maskChannels, maskSize * maskSize, CV_32F, prototypes.data());
  for (int index : selected) {
    const auto box = boxes[index] & cv::Rect2d(content);
    if (box.empty()) continue;
    cv::Mat coefficients(1, maskChannels, CV_32F);
    for (int channel = 0; channel < maskChannels; ++channel)
      coefficients.at<float>(0, channel) = detections[(5 + channel) * candidates + anchors[index]];
    cv::Mat logits = coefficients * prototypeMatrix;
    cv::Mat fullMask, sourceMask, binary;
    cv::resize(logits.reshape(1, maskSize), fullMask, inputSize, 0, 0, cv::INTER_LINEAR);
    cv::resize(fullMask(content), sourceMask, sourceSize, 0, 0, cv::INTER_LINEAR);
    const int left = int((box.x - content.x) * scaleX);
    const int top = int((box.y - content.y) * scaleY);
    const int right = std::min(sourceSize.width, int(std::ceil((box.br().x - content.x) * scaleX)));
    const int bottom = std::min(sourceSize.height, int(std::ceil((box.br().y - content.y) * scaleY)));
    cv::compare(sourceMask(cv::Rect(left, top, right - left, bottom - top)), 0, binary, cv::CMP_GT);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, {left, top});
    const auto polygon = std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
      return cv::contourArea(a) < cv::contourArea(b);
    });
    if (polygon == contours.end() || polygon->size() < 3) continue;
    if (!first) json << ',';
    first = false;
    json << "{\"x\":" << (box.x + box.width / 2 - content.x) * scaleX
         << ",\"y\":" << (box.y + box.height / 2 - content.y) * scaleY
         << ",\"width\":" << box.width * scaleX << ",\"height\":" << box.height * scaleY
         << ",\"confidence\":" << scores[index] << ",\"class\":\"plate\",\"class_id\":0,\"points\":[";
    for (size_t point = 0; point < polygon->size(); ++point) {
      if (point) json << ',';
      json << "{\"x\":" << (*polygon)[point].x << ",\"y\":" << (*polygon)[point].y << '}';
    }
    json << "]}";
  }
  json << "],\"image\":{\"width\":" << sourceSize.width << ",\"height\":" << sourceSize.height << "}}";
  return json.str();
}
