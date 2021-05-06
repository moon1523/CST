#ifndef INCLUDE_VTKFUNC_H_
#define INCLUDE_VTKFUNC_H_

#include <opencv2/viz.hpp>
#include <opencv2/viz/widget_accessor.hpp>

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPLYReader.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTransform.h>
#include <vtkActor.h>
#include <vtkPLYWriter.h>
#include "LabelData.h"

using namespace cv;

bool printPLY(false);

class WPoly : public viz::Widget3D
{
public:
    WPoly(){}
    WPoly(const string & fileName, const LabelData & label);
    vtkAlgorithmOutput* GetPolyDataPort() {return reader->GetOutputPort();}
    void Initialize(const string & fileName, const LabelData & label);
    void Transform(string recordFileName, int class_id, int template_id, double pos[3], int frameIdx);
private:
    LabelData label;
    vtkPLYReader* reader;
    vtkSmartPointer<vtkPolyDataMapper> mapper;
    vtkSmartPointer<vtkTransform> transform;
    vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter;
    vtkSmartPointer<vtkPLYWriter> plyWriter;
};

WPoly::WPoly(const string & fileName, const LabelData & _label)
{
    Initialize(fileName, _label);
}

void WPoly::Initialize(const string &fileName, const LabelData &_label){
    label = _label;
    transform = vtkSmartPointer<vtkTransform>::New();
    transformFilter =
            vtkSmartPointer<vtkTransformPolyDataFilter>::New();

    vtkSmartPointer<vtkPolyData> polyData;
    reader = vtkPLYReader::New ();
    reader->SetFileName (fileName.c_str());
    reader->Update ();
    polyData = reader->GetOutput ();
    // Create mapper and actor
    mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polyData);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Store this actor in the widget in order that visualizer can access it
    viz::WidgetAccessor::setProp(*this, actor);

    plyWriter = vtkSmartPointer<vtkPLYWriter>::New();
}

void WPoly::Transform(string pathPrint, int class_id, int template_id, double point[3], int frameIdx){
    transform->SetMatrix(label.GetAffineTransformMatrix(class_id, template_id, point));
    transformFilter->SetTransform(transform);
    transformFilter->SetInputConnection(reader->GetOutputPort());
    transformFilter->Update();
    mapper->SetInputData(transformFilter->GetOutput());

    if (printPLY) {
    	string fileName = pathPrint + "ply/frame" + to_string(frameIdx);
    	plyWriter->SetFileName(fileName.c_str());
    	plyWriter->SetInputData(transformFilter->GetOutput());
    	plyWriter->Write();
    }
}

#endif /* INCLUDE_VTKFUNC_H_ */
