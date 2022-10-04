#include "initialsetup.h"
#include "ui_initialsetup.h"

initialSetup::initialSetup(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::initialSetup)
{
    ui->setupUi(this);
    this->options = NULL;
}

initialSetup::~initialSetup()
{
    delete ui;
}

void initialSetup::acceptOptions(startupOptionsType *opts)
{
    this->options = opts;
//    if(options->xioDirectory == NULL)
//    {
//        options->xioDirectory = new QString();
//    }
    if(options->heightWidthSet)
    {
        ui->heightSpin->blockSignals(true);
        ui->widthSpin->blockSignals(true);
        ui->heightSpin->setValue(options->xioHeight);
        ui->widthSpin->setValue(options->xioWidth);
        ui->heightSpin->blockSignals(false);
        ui->widthSpin->blockSignals(false);
    }

//    if(options->xioDirectory == NULL)
//    {
//        abort(); // let's know if this happens.
//    }

//    if(!options->xioDirectory->isEmpty())
//    {
//      //  ui->xioPathText->setText(*options->xioDirectory);
//    }

    if(options->xioDirectoryArray == NULL)
    {
        abort();
    } else {
        ui->xioPathText->setText(options->xioDirectoryArray);

//        int c = 0;
//        while((c<4096) && options->xioDirectoryArray[c])
    }

    ui->fpsSpin->setValue((double)options->targetFPS);
}

void initialSetup::setHeightWidthLock(bool locked)
{
    this->lockHeightWidthControls = locked;
    ui->heightSpin->setEnabled(!locked);
    ui->widthSpin->setEnabled(!locked);
}

startupOptionsType *initialSetup::getOptions()
{
    return options;
}

void initialSetup::on_selectButton_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                 "/mnt",
                                                 QFileDialog::ShowDirsOnly
                                                 | QFileDialog::DontResolveSymlinks);
    if(dir.isEmpty())
    {
        return;
    } else {
        ui->xioPathText->setText(dir);
        if(options->xioDirectoryArray == NULL)
        {
            abort();
        } else {
            if (dir.length() > 4096)
                abort();
            int n=0;
            for(; n < dir.length(); n++)
            {
                options->xioDirectoryArray[n] = dir.toLocal8Bit().at(n);
            }
        }
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
//    if(ui->xioPathText->text().isEmpty())
//    {
//        if(!options->xioDirectory->isEmpty())
//        {
//            ui->xioPathText->setText(*options->xioDirectory);
//        } else {
//            // not sure, we do not have a good default here...
//        }
//        return;
//    } else {
//        options->xioDirectory->clear();
//        options->xioDirectory->append(ui->xioPathText->text());
//    }
}

void initialSetup::on_xioPathText_returnPressed()
{
//    if(options->xioDirectory == NULL)
//    {
//        abort();
//    }

//    if(ui->xioPathText->text().isEmpty())
//    {
//        return;
//    } else {

//        options->xioDirectory->clear();
//        options->xioDirectory->append(ui->xioPathText->text());
//    }
}

void initialSetup::on_buttonBox_rejected()
{
    // TODO: Do not abort, since we can now bring this box up
    // while the program is running.
    //abort();
}

void initialSetup::on_buttonBox_accepted()
{
    if((options->xioHeight !=0) && (options->xioWidth != 0))
        options->heightWidthSet = true;

//    if(options->xioDirectory == NULL)
//    {
//        abort();
//    }
//    if(options->xioDirectory->isEmpty())
//    {
//        abort();
//    }
    char c;
    int n=0;
    for(n=0; (n < ui->xioPathText->text().length()) and (n < 4094); n++)
    {
        c = ui->xioPathText->text().at(n).toLatin1();
        options->xioDirectoryArray[n] = c;
    }
    options->xioDirectoryArray[n] = '\x00';
}

void initialSetup::on_fpsSpin_valueChanged(double arg1)
{
    options->targetFPS = (float)arg1;
}
