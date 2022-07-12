#include "initialsetup.h"
#include "ui_initialsetup.h"

initialSetup::initialSetup(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::initialSetup)
{
    ui->setupUi(this);
}

initialSetup::~initialSetup()
{
    delete ui;
}

void initialSetup::acceptOptions(startupOptionsType *opts)
{
    this->options = opts;
    if(options->heightWidthSet)
    {
        ui->heightSpin->blockSignals(true);
        ui->widthSpin->blockSignals(true);
        ui->heightSpin->setValue(options->xioHeight);
        ui->widthSpin->setValue(options->xioWidth);
        ui->heightSpin->blockSignals(false);
        ui->widthSpin->blockSignals(false);
    }
    if(!options->xioDirectory.isEmpty())
    {
        ui->xioPathText->setText(options->xioDirectory);
    }
}

startupOptionsType *initialSetup::getOptions()
{
    return options;
}

void initialSetup::on_selectButton_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                 "/",
                                                 QFileDialog::ShowDirsOnly
                                                 | QFileDialog::DontResolveSymlinks);
    if(dir.isEmpty())
    {
        return;
    } else {
        ui->xioPathText->setText(dir);
        options->xioDirectory = dir;
    }
}

void initialSetup::on_heightSpin_valueChanged(int arg1)
{
    options->xioHeight = arg1;
}

void initialSetup::on_widthSpin_valueChanged(int arg1)
{
    options->xioWidth = arg1;
}

void initialSetup::on_xioPathText_editingFinished()
{
    if(ui->xioPathText->text().isEmpty())
    {
        if(!options->xioDirectory.isEmpty())
        {
            ui->xioPathText->setText(options->xioDirectory);
        } else {
            // not sure, we do not have a good default here...
        }
        return;
    } else {
        options->xioDirectory = ui->xioPathText->text();
    }
}

void initialSetup::on_xioPathText_returnPressed()
{
    if(ui->xioPathText->text().isEmpty())
    {
        return;
    } else {
        options->xioDirectory = ui->xioPathText->text();
    }
}

void initialSetup::on_buttonBox_rejected()
{
    abort();
}

void initialSetup::on_buttonBox_accepted()
{
    if(options->xioDirectory.isEmpty())
    {
        abort();
    }
}
