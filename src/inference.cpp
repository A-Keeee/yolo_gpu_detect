#include "inference.h"
#include <regex>

#define benchmark
#define min(a,b)            (((a) < (b)) ? (a) : (b))
YOLO_V8::YOLO_V8() {

}


YOLO_V8::~YOLO_V8() {
    delete session;
}

#ifdef USE_CUDA
namespace Ort
{
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif


template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;
}


char* YOLO_V8::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
{
    if (iImg.channels() == 3)
    {
        oImg = iImg.clone();
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    }
    else
    {
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }

    switch (modelType)
    {
    case YOLO_DETECT_V8:
    case YOLO_POSE:
    case YOLO_DETECT_V8_HALF:
    case YOLO_POSE_V8_HALF://LetterBox
    case YOLO_ARMOR:
    {
        if (iImg.cols >= iImg.rows)
        {
            resizeScales = iImg.cols / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
        }
        else
        {
            resizeScales = iImg.rows / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
        }
        cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
        oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
        oImg = tempImg;
        break;
    }
    case YOLO_CLS://CenterCrop
    {
        int h = iImg.rows;
        int w = iImg.cols;
        int m = min(h, w);
        int top = (h - m) / 2;
        int left = (w - m) / 2;
        cv::resize(oImg(cv::Rect(left, top, m, m)), oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)));
        break;
    }
    }
    return RET_OK;
}


char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams) {
    char* Ret = RET_OK;
    std::regex pattern("[\u4e00-\u9fa5]");
    bool result = std::regex_search(iParams.modelPath, pattern);
    if (result)
    {
        Ret = "[YOLO_V8]:Your model path is error.Change your model path without chinese characters.";
        std::cout << Ret << std::endl;
        return Ret;
    }
    try
    {
        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
        iouThreshold = iParams.iouThreshold;
        imgSize = iParams.imgSize;
        modelType = iParams.modelType;
        cudaEnable = iParams.cudaEnable;
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        Ort::SessionOptions sessionOption;
        if (iParams.cudaEnable)
        {
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

#ifdef _WIN32
        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
        wide_cstr[ModelPathSize] = L'\0';
        const wchar_t* modelPath = wide_cstr;
#else
        const char* modelPath = iParams.modelPath.c_str();
#endif // _WIN32

        session = new Ort::Session(env, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = session->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; i++)
        {
            Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames.push_back(temp_buf);
        }
        size_t OutputNodesNum = session->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; i++)
        {
            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
            char* temp_buf = new char[10];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames.push_back(temp_buf);
        }
        options = Ort::RunOptions{ nullptr };
        WarmUpSession();
        return RET_OK;
    }
    catch (const std::exception& e)
    {
        const char* str1 = "[YOLO_V8]:";
        const char* str2 = e.what();
        std::string result = std::string(str1) + std::string(str2);
        char* merged = new char[result.length() + 1];
        std::strcpy(merged, result.c_str());
        std::cout << merged << std::endl;
        delete[] merged;
        return "[YOLO_V8]:Create session failed.";
    }

}


double sigmoid(double x) 
{
    if(x>0)
        return 1.0 / (1.0 + exp(-x));
    else
        return exp(x) / (1.0 + exp(x));
}


char* YOLO_V8::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {
#ifdef benchmark
    clock_t starttime_1 = clock();
#endif // benchmark

    char* Ret = RET_OK;
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    if (modelType < 4)
    {
        float* blob = new float[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
    }
    else
    {
#ifdef USE_CUDA
        half* blob = new half[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1,3,imgSize.at(0),imgSize.at(1) };
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
#endif
    }

    return Ret;
}


template<typename N>
char* YOLO_V8::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
    std::vector<DL_RESULT>& oResult) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        inputNodeDims.data(), inputNodeDims.size());
#ifdef benchmark
    clock_t starttime_2 = clock();
#endif // benchmark
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
#ifdef benchmark
    clock_t starttime_3 = clock();
#endif // benchmark

    // std::cout << "outputTensor.size():" << outputTensor.size() << std::endl;
    // std::cout << "outputTensor.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().size():" << outputTensor.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().size() << std::endl;
    // std::cout << "outputTensor.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().at(0):" << outputTensor.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().at(0) << std::endl;
    // std::cout << "outputTensor.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().at(1):" << outputTensor.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().at(1) << std::endl;
    // std::cout << "outputTensor.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().at(2):" << outputTensor.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().at(2) << std::endl;

    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();
    delete[] blob;
    switch (modelType)
    {
    case YOLO_DETECT_V8:
    case YOLO_DETECT_V8_HALF:
    case YOLO_ARMOR:
    case YOLO_POSE:
    {
        int signalResultNum = outputNodeDims[1];//25200
        int strideNum = outputNodeDims[2];//22

        // std::cout << "signalResultNum:" << signalResultNum << std::endl;
        // std::cout << "strideNum:" << strideNum << std::endl;


        std::vector<int> class_ids;
        std::vector<int> color_ids;
        std::vector<int> number_ids;
        std::vector<float> confidences;
        std::vector<float> c_confidences;
        std::vector<float> n_confidences;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<cv::Point2f>> key_points_vec;
        cv::Mat rawData;
        if (modelType == YOLO_DETECT_V8 || modelType == YOLO_ARMOR || modelType == YOLO_POSE)
        {
            // FP32
            rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
        }
        else
        {
            // FP16
            rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }
        // Note:
        // ultralytics add transpose operator to the output of yolov8 model.which make yolov8/v5/v7 has same shape
        // https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

        // std::cout << "rawData:" << rawData << std::endl;


        if (modelType == YOLO_DETECT_V8 || modelType == YOLO_DETECT_V8_HALF){
            rawData = rawData.t();
            float* data = (float*)rawData.data;
            for (int i = 0; i < strideNum; ++i)
            {
                float* classesScores = data + 4;
                cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
                cv::Point class_id;
                double maxClassScore;
                cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
                if (maxClassScore > rectConfidenceThreshold)
                {   
                    confidences.push_back(maxClassScore);
                    class_ids.push_back(class_id.x);
                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w) * resizeScales);
                    int top = int((y - 0.5 * h) * resizeScales);

                    int width = int(w * resizeScales);
                    int height = int(h * resizeScales);

                    boxes.push_back(cv::Rect(left, top, width, height));
                }
                data += signalResultNum;
            }
            std::vector<int> nmsResult;
            cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
            for (int i = 0; i < nmsResult.size(); ++i)
            {
                int idx = nmsResult[i];
                DL_RESULT result;
                result.classId = class_ids[idx];
                result.confidence = confidences[idx];
                result.box = boxes[idx];
                oResult.push_back(result);
            }
        }
        else if(modelType == YOLO_ARMOR){
            // rawData = rawData.t();
            float* data = (float*)rawData.data;
            // std::cout << "YOLO_ARMOR" << std::endl;
            // std::cout << "signalResultNum:" << signalResultNum << std::endl;
            // std::cout << "strideNum:" << strideNum << std::endl;
            //0,8 xyxyxyxy 9,13 color 13,22 number
            for (int i = 0; i < signalResultNum; ++i)
                {   

                    float* colorScores = data + 9;
                    float* numberScores = data + 13;
                    cv::Mat c_scores(1, 4, CV_32FC1, colorScores);
                    cv::Mat n_scores(1, 9, CV_32FC1, numberScores);
                    cv::Point color_id;
                    cv::Point number_id;
                    double maxcolorScore;
                    double maxNumberScore;
                    cv::minMaxLoc(c_scores, 0, &maxcolorScore, 0, &color_id);
                    cv::minMaxLoc(n_scores, 0, &maxNumberScore, 0, &number_id);
                    if ((sigmoid(data[8])> rectConfidenceThreshold))
                    {   
                        // std::cout << "rectConfidenceThreshold:" << rectConfidenceThreshold << std::endl;
                        // std::cout << "data[8]:" << data[8] << std::endl;
                        // std::cout << "resizeScales:" << resizeScales << std::endl;
                        // std::cout << "maxcolorScore:" << maxcolorScore << std::endl;
                        // std::cout << "maxNumberScore:" << maxNumberScore << std::endl;
                        confidences.push_back(sigmoid(data[8]));
                        c_confidences.push_back(sigmoid(maxcolorScore));
                        n_confidences.push_back(sigmoid(maxNumberScore));
                        color_ids.push_back(color_id.x);
                        number_ids.push_back(number_id.x);
                        float x1 = data[0] * resizeScales;
                        float y1 = data[1] * resizeScales;
                        float x2 = data[2] * resizeScales;
                        float y2 = data[3] * resizeScales;
                        float x3 = data[4] * resizeScales;
                        float y3 = data[5] * resizeScales;
                        float x4 = data[6] * resizeScales;
                        float y4 = data[7] * resizeScales;
                        
                        std::vector<cv::Point2f> four_point;
                        four_point.push_back(cv::Point(x1, y1));
                        four_point.push_back(cv::Point(x4, y4));
                        four_point.push_back(cv::Point(x3, y3));
                        four_point.push_back(cv::Point(x2, y2));
                        key_points_vec.push_back(four_point);
                        // int left = int((x - 0.5 * w) * resizeScales);
                        // int top = int((y - 0.5 * h) * resizeScales);

                        // int width = int(w * resizeScales);
                        // int height = int(h * resizeScales);

                        boxes.push_back(cv::Rect(x1, y1, x3-x1, y3-y1));

                        // boxes

                        // std::cout << "x1:" << x1 << std::endl;
                        // std::cout << "y1:" << y1 << std::endl;
                    }
                    data += strideNum;
            }
            std::vector<int> nmsResult;
            cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
            for (int i = 0; i < nmsResult.size(); ++i)
            {
                int idx = nmsResult[i];
                DL_RESULT result;
                result.classId = number_ids[idx];
                result.confidence = confidences[idx];
                result.box = boxes[idx];
                std::vector<cv::Point2f> four_points = key_points_vec[idx];
                result.keyPoints = four_points; 
                oResult.push_back(result);
            }

        }

        else if(modelType == YOLO_POSE){
            std::cout << "YOLO_POSE" << std::endl;
            rawData = rawData.t();
            float* data = (float*)rawData.data;

            cv::waitKey(0);


            for (int i = 0; i < strideNum; ++i)
            {
                float* classesScores = data + 4;
                float* keyPoints = data + 8;
                cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
                cv::Point class_id;
                double maxClassScore;
                cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
                if (maxClassScore > rectConfidenceThreshold)
                {   
                    confidences.push_back(maxClassScore);
                    class_ids.push_back(class_id.x);
                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    float x1 = keyPoints[0] * resizeScales;
                    float y1 = keyPoints[1] * resizeScales;
                    float x2 = keyPoints[3] * resizeScales;
                    float y2 = keyPoints[4] * resizeScales;
                    float x3 = keyPoints[6] * resizeScales;
                    float y3 = keyPoints[7] * resizeScales;
                    float x4 = keyPoints[9] * resizeScales;
                    float y4 = keyPoints[10] * resizeScales;
                    float x5 = keyPoints[12] * resizeScales;
                    float y5 = keyPoints[13] * resizeScales;

                    std::vector<cv::Point2f> key_point;
                    key_point.push_back(cv::Point(x1, y1));
                    key_point.push_back(cv::Point(x2, y2));
                    key_point.push_back(cv::Point(x3, y3));
                    key_point.push_back(cv::Point(x4, y4));
                    key_point.push_back(cv::Point(x5, y5));
                    key_points_vec.push_back(key_point);

                    int left = int((x - 0.5 * w) * resizeScales);
                    int top = int((y - 0.5 * h) * resizeScales);

                    int width = int(w * resizeScales);
                    int height = int(h * resizeScales);

                    boxes.push_back(cv::Rect(left, top, width, height));
                }
                data += signalResultNum;
            }
            std::vector<int> nmsResult;
            cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
            for (int i = 0; i < nmsResult.size(); ++i)
            {
                int idx = nmsResult[i];
                DL_RESULT result;
                result.classId = class_ids[idx];
                result.confidence = confidences[idx];
                result.box = boxes[idx];
                std::vector<cv::Point2f> key_points = key_points_vec[idx];
                result.keyPoints = key_points; 
                oResult.push_back(result);
            }
        }





#ifdef benchmark
        clock_t starttime_4 = clock();
        double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
        double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
        double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
        }
        else
        {
            std::cout << "[YOLO_V8(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
        }
#endif // benchmark

        break;
    }
    case YOLO_CLS:
    case YOLO_CLS_HALF:
    {
        cv::Mat rawData;
        if (modelType == YOLO_CLS) {
            // FP32
            rawData = cv::Mat(1, this->classes.size(), CV_32F, output);
        } else {
            // FP16
            rawData = cv::Mat(1, this->classes.size(), CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }
        float *data = (float *) rawData.data;

        DL_RESULT result;
        for (int i = 0; i < this->classes.size(); i++)
        {
            result.classId = i;
            result.confidence = data[i];
            oResult.push_back(result);
        }
        break;
    }
    default:
        std::cout << "[YOLO_V8]: " << "Not support model type." << std::endl;
    }
    return RET_OK;

}


char* YOLO_V8::WarmUpSession() {
    clock_t starttime_1 = clock();
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    if (modelType < 4)
    {
        float* blob = new float[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
            YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
            outputNodeNames.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
    }
    else
    {
#ifdef USE_CUDA
        half* blob = new half[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = { 1,3,imgSize.at(0),imgSize.at(1) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<half>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), outputNodeNames.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
#endif
    }
    return RET_OK;
}