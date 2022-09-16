#ifndef YOLOV4_H
#define YOLOV4_H

#include "YoloV4Global.h"
#include "Process/OpenCV/dnn/COcvDnnProcess.h"
#include "Widget/OpenCV/dnn/COcvWidgetDnnCore.h"
#include "CPluginProcessInterface.hpp"

//------------------------//
//----- CYoloV4Param -----//
//------------------------//
class YOLOV4SHARED_EXPORT CYoloV4Param: public COcvDnnProcessParam
{
    public:

        CYoloV4Param();

        void        setParamMap(const UMapString& paramMap) override;

        UMapString  getParamMap() const override;

    public:

        std::string m_modelFolder;
        double      m_confidence = 0.5;
        double      m_nmsThreshold = 0.4;
};

//-------------------//
//----- CYoloV4 -----//
//-------------------//
class YOLOV4SHARED_EXPORT CYoloV4: public COcvDnnProcess
{
    public:

        CYoloV4();
        CYoloV4(const std::string& name, const std::shared_ptr<CYoloV4Param>& pParam);

        size_t      getProgressSteps() override;
        int         getNetworkInputSize() const override;
        double      getNetworkInputScaleFactor() const override;
        cv::Scalar  getNetworkInputMean() const override;

        void        run() override;

    private:

        void        manageOutput(const std::vector<cv::Mat> &dnnOutputs);
        void        generateColors();

    private:

        std::vector<CColor> m_colors;
};

//--------------------------//
//----- CYoloV4Factory -----//
//--------------------------//
class YOLOV4SHARED_EXPORT CYoloV4Factory : public CTaskFactory
{
    public:

        CYoloV4Factory()
        {
            m_info.m_name = "infer_yolo_v4";
            m_info.m_shortDescription = QObject::tr("Object detection using YOLO V4 neural network").toStdString();
            m_info.m_description = QObject::tr("There are a huge number of features which are said to improve Convolutional Neural Network (CNN) accuracy."
                                               "Practical testing of combinations of such features on large datasets, and theoretical justification  of  "
                                               "the result, is required. Some features operate on certain models exclusively and for certain problems exclusively, "
                                               "or only for small-scale datasets; while some features, such as batch-normalization and residual-connections, "
                                               "are applicable to the majority of models, tasks, and datasets. We assume that such universal features include "
                                               "Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), Cross mini-Batch Normalization (CmBN), "
                                               "Self-adversarial-training (SAT) and Mish-activation. We use new features: WRC, CSP, CmBN, SAT, Mish activation, "
                                               "Mosaic data augmentation, CmBN, DropBlock regularization, and CIoU loss, and combine some of them to achieve "
                                               "state-of-the-art results: 43.5%AP (65.7% AP50) for the MS COCO dataset at a real-time speed of ∼65 FPS on Tesla V100.").toStdString();
            m_info.m_path = QObject::tr("Plugins/C++/Detection").toStdString();
            m_info.m_version = "1.3.0";
            m_info.m_iconPath = "Icon/icon.png";
            m_info.m_authors = "Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao";
            m_info.m_article = "YOLOv4: Optimal Speed and Accuracy of Object Detection";
            m_info.m_year = 2020;
            m_info.m_license = "YOLO License (public)";
            m_info.m_repo = "https://github.com/AlexeyAB/darknet";
            m_info.m_keywords = "deep,learning,detection,yolo,darknet," + Utils::Plugin::getArchitectureKeywords();
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto paramPtr = std::dynamic_pointer_cast<CYoloV4Param>(pParam);
            if(paramPtr != nullptr)
                return std::make_shared<CYoloV4>(m_info.m_name, paramPtr);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto paramPtr = std::make_shared<CYoloV4Param>();
            assert(paramPtr != nullptr);
            return std::make_shared<CYoloV4>(m_info.m_name, paramPtr);
        }
};

//-------------------------//
//----- CYoloV4Widget -----//
//-------------------------//
class YOLOV4SHARED_EXPORT CYoloV4Widget: public COcvWidgetDnnCore
{
    public:

        CYoloV4Widget(QWidget *parent = Q_NULLPTR);
        CYoloV4Widget(WorkflowTaskParamPtr pParam, QWidget *parent = Q_NULLPTR);

        void onApply() override;

    private:

        void init();

    private:

        QSpinBox*           m_pSpinInputSize = nullptr;
        QComboBox*          m_pComboModel = nullptr;
        QComboBox*          m_pComboDataset = nullptr;
        CBrowseFileWidget*  m_pBrowseConfig = nullptr;
        CBrowseFileWidget*  m_pBrowseWeights = nullptr;
        CBrowseFileWidget*  m_pBrowseLabels = nullptr;
};

//--------------------------------//
//----- CYoloV4WidgetFactory -----//
//--------------------------------//
class YOLOV4SHARED_EXPORT CYoloV4WidgetFactory : public CWidgetFactory
{
    public:

        CYoloV4WidgetFactory()
        {
            m_name = "infer_yolo_v4";
        }

        virtual WorkflowTaskWidgetPtr   create(WorkflowTaskParamPtr pParam)
        {
            return std::make_shared<CYoloV4Widget>(pParam);
        }
};

//-----------------------------------//
//----- Global plugin interface -----//
//-----------------------------------//
class YOLOV4SHARED_EXPORT CYoloV4Interface : public QObject, public CPluginProcessInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "ikomia.plugin.process")
    Q_INTERFACES(CPluginProcessInterface)

    public:

        virtual std::shared_ptr<CTaskFactory>   getProcessFactory()
        {
            return std::make_shared<CYoloV4Factory>();
        }

        virtual std::shared_ptr<CWidgetFactory> getWidgetFactory()
        {
            return std::make_shared<CYoloV4WidgetFactory>();
        }
};

#endif // YOLOV4_H
