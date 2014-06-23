#include "controlsbox.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QFileDialog>
ControlsBox::ControlsBox(QTabWidget *tw, QWidget *parent) :
    QGroupBox(parent)
{
    QHBoxLayout * controls_layout =new QHBoxLayout;
    cur_frameview=NULL;
    qtw = tw;
    //Collection Buttons

        CollectionButtonsBox =new QWidget();
        QGridLayout * collections_layout = new QGridLayout();

        run_collect_button = new QPushButton("Run Collect");
        run_display_button = new QPushButton("Run Display");
        stop_collect_button = new QPushButton("Stop Collect");
        stop_display_button = new QPushButton("Stop Display");

        collect_dark_frames_button = new QPushButton("Collect Dark Frames");
        stop_dark_collection_button = new QPushButton("Stop Dark Collection");

        load_mask_from_file = new QPushButton("Load Mask From File");
        fps_label = new QLabel("Warning: no data recieved");

        //First Row
        collections_layout->addWidget(run_display_button,1,1,1,2);
        //collections_layout->addWidget(run_display_button,1,2);
        collections_layout->addWidget(stop_display_button,1,3,1,2);
        //collections_layout->addWidget(stop_display_button,1,4);

        //Second Row
        collections_layout->addWidget(collect_dark_frames_button,2,1,1,2);
        collections_layout->addWidget(stop_dark_collection_button,2,3,1,1);
        collections_layout->addWidget(load_mask_from_file,2,4,1,1);

        //Third Row
        collections_layout->addWidget(fps_label,3,1,1,4);

        CollectionButtonsBox->setLayout(collections_layout);

    //Slider Thresholding Buttons
    ThresholdingSlidersBox = new QWidget();
    QGridLayout * sliders_layout = new QGridLayout();
    show_dark_subtracted_cbox = new QCheckBox("Show Dark Subtracted");
    ceiling_slider = new QSlider(Qt::Horizontal);
    ceiling_slider->setMaximum(max);
    ceiling_slider->setMinimum(min);
    floor_slider = new QSlider(Qt::Horizontal);
    floor_slider->setMaximum(max);
    floor_slider->setMinimum(min);
    ceiling_edit = new QSpinBox();
    ceiling_edit->setMaximum(max);
    ceiling_edit->setMinimum(min);
    ceiling_edit->setButtonSymbols(QAbstractSpinBox::NoButtons);

    floor_edit = new QSpinBox();
    floor_edit->setMaximum(max);
    floor_edit->setMinimum(min);
    floor_edit->setButtonSymbols(QAbstractSpinBox::NoButtons);


    //First Row
    sliders_layout->addWidget(show_dark_subtracted_cbox,1,1,1,1);

    //Second Row
    sliders_layout->addWidget(new QLabel("Ceiling:"),2,1,1,1);
    sliders_layout->addWidget(ceiling_slider,2,2,1,3);
    sliders_layout->addWidget(ceiling_edit,2,5,1,1);

    //Third Row
    sliders_layout->addWidget(new QLabel("Window:"),3,1,1,1);
    sliders_layout->addWidget(floor_slider,3,2,1,3);
    sliders_layout->addWidget(floor_edit,3,5,1,1);
    ThresholdingSlidersBox->setLayout(sliders_layout);

    //Save Buttons
    SaveButtonsBox = new QWidget();

    save_frame_button = new QPushButton("Save Frame");
    save_dark_button = new QPushButton("Save Dark");

    save_frames_button = new QPushButton("Save Frames");
    stop_saving_frames_button = new QPushButton("Stop Saving");
    frames_save_num_edit = new QSpinBox();
    frames_save_num_edit->setButtonSymbols(QAbstractSpinBox::NoButtons);
    filename_edit = new QLineEdit();
    set_filename_button = new QPushButton();


    QGridLayout * save_layout = new QGridLayout();
    QGroupBox * single_save_box = new QGroupBox();
    QVBoxLayout * single_save_layout = new QVBoxLayout;
#include "frameview_widget.h"

    single_save_layout->addWidget(new QLabel("Single Save"));
    single_save_layout->addWidget(save_frame_button);
    single_save_layout->addWidget(save_dark_button);

    single_save_box->setLayout(single_save_layout);


    save_layout->addWidget(single_save_box,1,1,3,1);

    //Column 2
    save_layout->addWidget(save_frames_button,1,2,1,1);
    save_layout->addWidget(new QLabel("Frames to save:"),2,2,1,1);
    save_layout->addWidget(new QLabel("Filename:"),3,2,1,1);

    //Column 3
    save_layout->addWidget(stop_saving_frames_button,1,3,1,1);
    save_layout->addWidget(frames_save_num_edit,2,3,1,1);
    save_layout->addWidget(filename_edit,3,3,1,1);
    SaveButtonsBox->setLayout(save_layout);

    controls_layout->addWidget(CollectionButtonsBox);
    controls_layout->addWidget(ThresholdingSlidersBox);
    controls_layout->addWidget(SaveButtonsBox);
    this->setLayout(controls_layout);
    this->setMaximumHeight(150);

    connect(this->ceiling_edit,SIGNAL(valueChanged(int)),this->ceiling_slider,SLOT(setValue(int)));
    connect(this->ceiling_slider,SIGNAL(valueChanged(int)),this->ceiling_edit,SLOT(setValue(int)));

    connect(this->floor_edit,SIGNAL(valueChanged(int)),this->floor_slider,SLOT(setValue(int)));
    connect(this->floor_slider,SIGNAL(valueChanged(int)),this->floor_edit,SLOT(setValue(int)));

    connect(this->load_mask_from_file,SIGNAL(clicked()),this,SLOT(getMaskFile()));
    connect(this->save_frames_button,SIGNAL(clicked()),this,SLOT(save_button_slot()));

}
void ControlsBox::getMaskFile()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Select mask file"),
                                                     "",
                                                     tr("Files (*.raw)"));
    if(fileName.isEmpty())
    {
        return;
    }
    std::string utf8_text = fileName.toUtf8().constData();
    emit mask_selected(utf8_text.c_str());

}

void ControlsBox::save_button_slot()
{
    emit startSaving(filename_edit->text().toLocal8Bit().data());
}
void ControlsBox::tabChangedSlot(int index)
{
    if(cur_frameview != NULL)
   {
        disconnect(ceiling_slider, SIGNAL(valueChanged(int)), cur_frameview, SLOT(updateCeiling(int)));
     disconnect(floor_slider, SIGNAL(valueChanged(int)), cur_frameview, SLOT(updateFloor(int)));
    }
    qDebug() << index;
   // qDebug() << qtw->count();
   cur_frameview = (frameview_widget * )qtw->widget(index);

    this->ceiling_edit->setValue(cur_frameview->getCeiling());
    this->floor_edit->setValue(cur_frameview->getFloor());
    connect(ceiling_slider, SIGNAL(valueChanged(int)), cur_frameview, SLOT(updateCeiling(int)));
    connect(floor_slider, SIGNAL(valueChanged(int)), cur_frameview, SLOT(updateFloor(int)));

}
