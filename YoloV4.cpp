#include "YoloV4.h"
#include "IO/CObjectDetectionIO.h"

//------------------------//
//----- CYoloV4Param -----//
//------------------------//
CYoloV4Param::CYoloV4Param() : COcvDnnProcessParam()
{
    m_framework = Framework::DARKNET;
    m_inputSize = 416;
    m_modelName = "YOLOv4";
    m_datasetName = "COCO";
    m_modelFolder = Utils::Plugin::getCppPath() + "/infer_yolo_v4/Model/";
    m_labelsFile = m_modelFolder + "coco_names.txt";
    m_structureFile = m_modelFolder + "yolov4.cfg";
    m_modelFile = m_modelFolder + "yolov4.weights";
}

void CYoloV4Param::setParamMap(const UMapString &paramMap)
{
    COcvDnnProcessParam::setParamMap(paramMap);
    m_confidence = std::stod(paramMap.at("confidence"));
    m_nmsThreshold = std::stod(paramMap.at("nmsThreshold"));
}

UMapString CYoloV4Param::getParamMap() const
{
    auto paramMap = COcvDnnProcessParam::getParamMap();
    paramMap.insert(std::make_pair("confidence", std::to_string(m_confidence)));
    paramMap.insert(std::make_pair("nmsThreshold", std::to_string(m_nmsThreshold)));
    return paramMap;
}

//-------------------//
//----- CYoloV4 -----//
//-------------------//
CYoloV4::CYoloV4() : COcvDnnProcess()
{
    m_pParam = std::make_shared<CYoloV4Param>();
    addOutput(std::make_shared<CObjectDetectionIO>());
}

CYoloV4::CYoloV4(const std::string &name, const std::shared_ptr<CYoloV4Param> &pParam): COcvDnnProcess(name)
{
    m_pParam = std::make_shared<CYoloV4Param>(*pParam);
    addOutput(std::make_shared<CObjectDetectionIO>());
}

size_t CYoloV4::getProgressSteps()
{
    return 3;
}

int CYoloV4::getNetworkInputSize() const
{
    auto pParam = std::dynamic_pointer_cast<CYoloV4Param>(m_pParam);
    int size = pParam->m_inputSize;

    // Trick to overcome OpenCV issue around CUDA context and multithreading
    // https://github.com/opencv/opencv/issues/20566
    if(pParam->m_backend == cv::dnn::DNN_BACKEND_CUDA && m_bNewInput)
        size = size + (m_sign * 32);

    return size;
}

double CYoloV4::getNetworkInputScaleFactor() const
{
    return 1.0 / 255.0;
}

cv::Scalar CYoloV4::getNetworkInputMean() const
{
    return cv::Scalar();
}

void CYoloV4::run()
{
    beginTaskRun();
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    auto pParam = std::dynamic_pointer_cast<CYoloV4Param>(m_pParam);

    if(pInput == nullptr || pParam == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    if(pInput->isDataAvailable() == false)
        throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

    CMat imgSrc;
    CMat imgOrigin = pInput->getImage();
    std::vector<cv::Mat> netOutputs;

    //Detection networks need color image as input
    if(imgOrigin.channels() < 3)
        cv::cvtColor(imgOrigin, imgSrc, cv::COLOR_GRAY2RGB);
    else
        imgSrc = imgOrigin;

    emit m_signalHandler->doProgress();

    try
    {
        if(m_net.empty() || pParam->m_bUpdate)
        {
            m_net = readDnn();
            if(m_net.empty())
                throw CException(CoreExCode::INVALID_PARAMETER, "Failed to load network", __func__, __FILE__, __LINE__);

            if(m_classNames.empty())
                readClassNames();

            generateColors();
            pParam->m_bUpdate = false;
        }
        forward(imgSrc, netOutputs);
    }
    catch(cv::Exception& e)
    {
        throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
    }

    endTaskRun();
    emit m_signalHandler->doProgress();
    manageOutput(netOutputs);
    emit m_signalHandler->doProgress();
}

void CYoloV4::manageOutput(const std::vector<cv::Mat>& dnnOutputs)
{
    forwardInputImage();

    auto pParam = std::dynamic_pointer_cast<CYoloV4Param>(m_pParam);
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    CMat imgSrc = pInput->getImage();

    const size_t nbClasses = m_classNames.size();
    std::vector<std::vector<cv::Rect2d>> boxes;
    std::vector<std::vector<float>> scores;
    std::vector<std::vector<int>> indices;
    boxes.resize(nbClasses);
    scores.resize(nbClasses);
    indices.resize(nbClasses);

    auto objDetectIOPtr = std::dynamic_pointer_cast<CObjectDetectionIO>(getOutput(1));
    objDetectIOPtr->init(getName(), 0);
    const int probabilityIndex = 5;

    for(auto&& output : dnnOutputs)
    {
        const auto nbBoxes = output.rows;
        for(int i=0; i<nbBoxes; ++i)
        {
            for(size_t j=0; j<nbClasses; ++j)
            {
                float confidence = output.at<float>(i, (int)j + probabilityIndex);
                if (confidence > pParam->m_confidence)
                {
                    float xCenter = output.at<float>(i, 0) * imgSrc.cols;
                    float yCenter = output.at<float>(i, 1) * imgSrc.rows;
                    float width = output.at<float>(i, 2) * imgSrc.cols;
                    float height = output.at<float>(i, 3) * imgSrc.rows;
                    float left = xCenter - width/2;
                    float top = yCenter - height/2;
                    cv::Rect2d r(left, top, width, height);
                    boxes[j].push_back(r);
                    scores[j].push_back(confidence);
                }
            }
        }
    }

    // Apply non-maximum suppression
    for(size_t i=0; i<nbClasses; ++i)
        cv::dnn::NMSBoxes(boxes[i], scores[i], pParam->m_confidence, pParam->m_nmsThreshold, indices[i]);

    int id = 0;
    for(size_t i=0; i<nbClasses; ++i)
    {
        for(size_t j=0; j<indices[i].size(); ++j)
        {
            const int index = indices[i][j];
            cv::Rect2d box = boxes[i][index];
            float confidence = scores[i][index];
            objDetectIOPtr->addObject(id++, m_classNames[i], confidence, box.x, box.y, box.width, box.height, m_colors[i]);
        }
    }
}

void CYoloV4::generateColors()
{
    //Random colors
    for(size_t i=0; i<m_classNames.size(); ++i)
    {
        m_colors.push_back({ (uchar)((double)std::rand() / (double)RAND_MAX * 255.0),
                             (uchar)((double)std::rand() / (double)RAND_MAX * 255.0),
                             (uchar)((double)std::rand() / (double)RAND_MAX * 255.0),
                           });
    }
}

//-------------------------//
//----- CYoloV4Widget -----//
//-------------------------//
CYoloV4Widget::CYoloV4Widget(QWidget *parent): COcvWidgetDnnCore(parent)
{
    init();
}

CYoloV4Widget::CYoloV4Widget(WorkflowTaskParamPtr pParam, QWidget *parent): COcvWidgetDnnCore(pParam, parent)
{
    m_pParam = std::dynamic_pointer_cast<CYoloV4Param>(pParam);
    init();
}

void CYoloV4Widget::init()
{
    if(m_pParam == nullptr)
        m_pParam = std::make_shared<CYoloV4Param>();

    auto pParam = std::dynamic_pointer_cast<CYoloV4Param>(m_pParam);
    assert(pParam);

    m_pSpinInputSize = addSpin(tr("Input size"), pParam->m_inputSize, 32, 2048, 32);

    m_pComboModel = addCombo(tr("Model"));
    m_pComboModel->addItem("YOLOv4x-mish");
    m_pComboModel->addItem("YOLOv4-csp");
    m_pComboModel->addItem("YOLOv4");
    m_pComboModel->addItem("Tiny YOLOv4");
    m_pComboModel->setCurrentText(QString::fromStdString(pParam->m_modelName));

    m_pComboDataset = addCombo(tr("Trained on"));
    m_pComboDataset->addItem("COCO");
    m_pComboDataset->addItem("Custom");
    m_pComboDataset->setCurrentText(QString::fromStdString(pParam->m_datasetName));

    m_pBrowseConfig = addBrowseFile(tr("Configuration file"), QString::fromStdString(pParam->m_structureFile), "");
    m_pBrowseConfig->setEnabled(pParam->m_datasetName == "Custom");

    m_pBrowseWeights = addBrowseFile(tr("Weights file"), QString::fromStdString(pParam->m_modelFile), "");
    m_pBrowseWeights->setEnabled(pParam->m_datasetName == "Custom");

    m_pBrowseLabels = addBrowseFile(tr("Labels file"), QString::fromStdString(pParam->m_labelsFile), "");
    m_pBrowseLabels->setEnabled(pParam->m_datasetName == "Custom");

    auto pSpinConfidence = addDoubleSpin(tr("Confidence"), pParam->m_confidence, 0.0, 1.0, 0.1, 2);
    auto pSpinNmsThreshold = addDoubleSpin(tr("NMS threshold"), pParam->m_nmsThreshold, 0.0, 1.0, 0.1, 2);

    //Connections
    connect(m_pComboModel, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int index)
    {
        Q_UNUSED(index);
        auto pParam = std::dynamic_pointer_cast<CYoloV4Param>(m_pParam);
        assert(pParam);
        pParam->m_modelName = m_pComboModel->currentText().toStdString();
        pParam->m_bUpdate = true;
    });
    connect(m_pComboDataset, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int index)
    {
        Q_UNUSED(index);
        auto pParam = std::dynamic_pointer_cast<CYoloV4Param>(m_pParam);
        assert(pParam);
        pParam->m_datasetName = m_pComboDataset->currentText().toStdString();
        m_pBrowseConfig->setEnabled(pParam->m_datasetName == "Custom");
        m_pBrowseWeights->setEnabled(pParam->m_datasetName == "Custom");
        m_pBrowseLabels->setEnabled(pParam->m_datasetName == "Custom");
        pParam->m_bUpdate = true;
    });
    connect(pSpinConfidence, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
    {
        auto pParam = std::dynamic_pointer_cast<CYoloV4Param>(m_pParam);
        assert(pParam);
        pParam->m_confidence = val;
    });
    connect(pSpinNmsThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
    {
        auto pParam = std::dynamic_pointer_cast<CYoloV4Param>(m_pParam);
        assert(pParam);
        pParam->m_nmsThreshold = val;
    });
}

void CYoloV4Widget::onApply()
{
    auto pParam = std::dynamic_pointer_cast<CYoloV4Param>(m_pParam);
    assert(pParam);
    pParam->m_inputSize = m_pSpinInputSize->value();

    if(pParam->m_datasetName == "COCO")
    {
        pParam->m_labelsFile = pParam->m_modelFolder + "coco_names.txt";
        m_pBrowseLabels->setPath(QString::fromStdString(pParam->m_labelsFile));

        if(pParam->m_modelName == "YOLOv4")
        {
            pParam->m_structureFile = pParam->m_modelFolder + "yolov4.cfg";
            pParam->m_modelFile = pParam->m_modelFolder + "yolov4.weights";
            m_pBrowseConfig->setPath(QString::fromStdString(pParam->m_structureFile));
            m_pBrowseWeights->setPath(QString::fromStdString(pParam->m_modelFile));
        }
        else if(pParam->m_modelName == "Tiny YOLOv4")
        {
            pParam->m_structureFile = pParam->m_modelFolder + "yolov4-tiny.cfg";
            pParam->m_modelFile = pParam->m_modelFolder + "yolov4-tiny.weights";
            m_pBrowseConfig->setPath(QString::fromStdString(pParam->m_structureFile));
            m_pBrowseWeights->setPath(QString::fromStdString(pParam->m_modelFile));
        }
        else if(pParam->m_modelName == "YOLOv4-csp")
        {
            pParam->m_structureFile = pParam->m_modelFolder + "yolov4-csp.cfg";
            pParam->m_modelFile = pParam->m_modelFolder + "yolov4-csp.weights";
            m_pBrowseConfig->setPath(QString::fromStdString(pParam->m_structureFile));
            m_pBrowseWeights->setPath(QString::fromStdString(pParam->m_modelFile));
        }
        else if(pParam->m_modelName == "YOLOv4x-mish")
        {
            pParam->m_structureFile = pParam->m_modelFolder + "yolov4x-mish.cfg";
            pParam->m_modelFile = pParam->m_modelFolder + "yolov4x-mish.weights";
            m_pBrowseConfig->setPath(QString::fromStdString(pParam->m_structureFile));
            m_pBrowseWeights->setPath(QString::fromStdString(pParam->m_modelFile));
        }
    }
    else
    {
        pParam->m_structureFile = m_pBrowseConfig->getPath().toStdString();
        pParam->m_modelFile = m_pBrowseWeights->getPath().toStdString();
        pParam->m_labelsFile = m_pBrowseLabels->getPath().toStdString();
    }
    emit doApplyProcess(m_pParam);
}
