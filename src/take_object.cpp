/*
 * takeobject.cpp
 *
 *  Created on: May 27, 2014
 *      Author: nlevy
 */

#include "take_object.hpp"
#include "fft.hpp"
//#define RESET_GPUS

take_object::take_object(int channel_num, int number_of_buffers, int fmsize, int frf)
{
    this->channel = channel_num;
    this->numbufs = number_of_buffers;
    this->filter_refresh_rate =frf;
    this->count = 0;

    this->std_dev_filter_N = 400;

    whichFFT = PLANE_MEAN;

    this->do_raw_save = false;

    dsfMaskCollected = false;
    frame_ring_buffer = new frame_c[CPU_FRAME_BUFFER_SIZE];
    save_framenum = 0;
    saving_list.clear();
    pdv_thread_run = 0;
}
take_object::~take_object()
{
    if(pdv_thread_run!=0)
    {
        int dummy;
        pdv_thread_run = 0;
        pdv_thread.join(); //Wait for thread to end
        pdv_wait_last_image(pdv_p,&dummy); //Collect the last frame to avoid core dump
        pdv_close(pdv_p);
        usleep(1000000);
#ifdef VERBOSE
        printf("about to delete filters!\n");
#endif
        delete dsf;
        delete sdvf;
    }
    delete[] frame_ring_buffer;


#ifdef RESET_GPUS
    printf("reseting GPUs!\n");
    int count;
    cudaGetDeviceCount(&count);
    for(int i = 0; i < count; i++)
    {
        printf("resetting GPU#%i",i);
        cudaSetDevice(i);
        cudaDeviceReset(); //Dump all are bad stuff from each of our GPUs.
    }
#endif
}


void take_object::start()
{
    pdv_thread_run = 1;

    std::cout << "This version of cuda_take was compiled on " << __DATE__ << " at " << __TIME__ << " using gcc " << __GNUC__ << std::endl;
    std::cout << "The compilation was perfromed by " << UNAME << " @ " << HOST << std::endl;

    this->pdv_p = NULL;
    this->pdv_p = pdv_open_channel(EDT_INTERFACE,0,this->channel);
    if(pdv_p == NULL)
    {
#ifdef VERBOSE
        std::cout << "Oh gawd, couldn't open channel" << std::endl;
#endif
        //TODO: throw exception
        return;
    }
    size = pdv_get_dmasize(pdv_p); //Not using this at the moment
    switch(size)
    {
    case 481*640*sizeof(uint16_t): cam_type = CL_6604A; break;
    case 285*640*sizeof(uint16_t): cam_type = CL_6604A; break;
    default: cam_type = CL_6604B; break;
    }
#ifdef VERBOSE
    printf("frame period:%i\n", pdv_get_frame_period(pdv_p));
#endif
    frWidth = pdv_get_width(pdv_p);
    //Our version of height should not include the header size
    dataHeight = pdv_get_height(pdv_p);
    frHeight = cam_type == CL_6604A ? dataHeight -1 : dataHeight;
#ifdef VERBOSE
    printf("cam type: %u. Width: %u Height %u frame height %u \n", cam_type, frWidth, dataHeight, frHeight);
#endif

    // initial dimensions for calculating the mean that can be updated later
    meanStartRow = 0;
    meanStartCol = 0;
    meanHeight = frHeight;
    meanWidth = frWidth;

    if( cam_type == CL_6604B )
        chromaPix = true;
#ifdef VERBOSE
    std::cout << "about to start threads" << std::endl;
#endif
    pdv_multibuf(pdv_p,this->numbufs);

    dsf = new dark_subtraction_filter(frWidth,frHeight);
    sdvf = new std_dev_filter(frWidth,frHeight);
    //numbufs = 16;
    pdv_start_images(pdv_p,numbufs); //Before looping, emit requests to fill the pdv ring buffer
    pdv_thread = boost::thread(&take_object::pdv_loop, this);
    //save_thread = boost::thread(&take_object::savingLoop, this);

}
void take_object::pdv_loop() //Producer Thread
{

    uint16_t framecount = 1;
    uint16_t last_framecount = 0;
    count = 0;

    u_char * wait_ptr;

    while(pdv_thread_run == 1)
    {
        //curFrame = std::shared_ptr<frame_c>(new frame_c());
        curFrame = &frame_ring_buffer[count % CPU_FRAME_BUFFER_SIZE];
        curFrame->reset();
        wait_ptr = pdv_wait_image(pdv_p);
        memcpy(curFrame->raw_data_ptr,wait_ptr,frWidth*dataHeight*sizeof(uint16_t));
        /* In this section of the code, after we have copied the memory from the camera link
         * buffer into the raw_dataptr, we will check various parameters to see if we need to
         * modify the data based on our hardware.
         *
         * First, the data is stored differently depending on the type of camera, 6604A or B.
         *
         * Second, we may have to offset the data's first or last row if it contains metadata.
         * This feature is controlled from the preference window in liveview.
         *
         * Third, we may need to invert the data range if a cable is inverting the magnitudes
         * that arrive from the ADC. This feature is also modified from the preference window.
         */
        if( chromaPix )
        {
            apply_chroma_translate_filter(curFrame->raw_data_ptr);
            // curFrame->image_data_ptr = curFrame->raw_data_ptr;
        }
        if(cam_type == CL_6604A)
        {
            curFrame->image_data_ptr = curFrame->raw_data_ptr + frWidth;
        }
        else
        {
            curFrame->image_data_ptr = curFrame->raw_data_ptr;
        }

        if( inverted )
        { // record the data from high to low. Store the pixel buffer in INVERTED order from the camera link
            for( uint i = 0; i < frHeight*frWidth; i++ )
                curFrame->image_data_ptr[i] = invFactor - curFrame->image_data_ptr[i];
        }

        sdvf->update_GPU_buffer(curFrame,std_dev_filter_N);

        dsf->update(curFrame->raw_data_ptr,curFrame->dark_subtracted_data);

        mean_filter * mf = new mean_filter(curFrame,count,meanStartCol,meanWidth,meanStartRow,meanHeight,frWidth,useDSF,whichFFT);
        //This will deallocate itself when it is done.
        mf->start_mean();

        if(save_framenum > 0)
        {
            uint16_t * raw_copy = new uint16_t[frWidth*dataHeight];
            memcpy(raw_copy,curFrame->raw_data_ptr,frWidth*dataHeight*sizeof(uint16_t));
            saving_list.push_front(raw_copy);
            save_framenum--;
        }
        pdv_start_image(pdv_p); //Start another

        framecount = *(curFrame->raw_data_ptr + 160); // wtf
        if(CHECK_FOR_MISSED_FRAMES_6604A && cam_type == CL_6604A)
        {
            if( framecount -1 != last_framecount)
            {
                std::cerr << "WARNING: MISSED FRAME " << framecount << std::endl;
            }
        }
        last_framecount = framecount;
        count++;
    }
}
/*void take_object::saveFramesInBuffer()
{
    register int ndx = count % CPU_FRAME_BUFFER_SIZE;
    // So! We need to make a copy of the frame ring buffer so that we can
    // take the raws out of the frames. So we need to copy starting from the current framecount
    // to the end of the array then from the beginning of the array to the current framecount.
    //
    frame_c* buf_copy = new frame_c[CPU_FRAME_BUFFER_SIZE];
    memcpy(buf_copy + ndx*sizeof(frame_c),frame_ring_buffer + ndx*sizeof(frame_c),(CPU_FRAME_BUFFER_SIZE - ndx)*sizeof(frame_c));
    memcpy(buf_copy,frame_ring_buffer,(ndx -1)*sizeof(frame_c));
    // we need to do one iteration of the saving loop first so that it doesn't immediately end the while loop...
    uint16_t raw_copy[frWidth*frHeight];
    memcpy(raw_copy,(buf_copy[ndx++]).raw_data_ptr,frWidth*dataHeight*sizeof(uint16_t));
    saving_list.push_front(raw_copy);
    while( ndx != (int)(count % CPU_FRAME_BUFFER_SIZE) )
    {
        uint16_t raw_copy[frWidth*frHeight];
        memcpy(raw_copy,(buf_copy[ndx++]).raw_data_ptr,frWidth*dataHeight*sizeof(uint16_t));
        saving_list.push_front(raw_copy);
        if( ndx == CPU_FRAME_BUFFER_SIZE )
            ndx = 0;
    }
    delete[] buf_copy;
}*/
void take_object::savingLoop(std::string fname)
{
    FILE * file_target = fopen(fname.c_str(), "wb");

    while(save_framenum != 0 || !saving_list.empty())
    {
        if(!saving_list.empty())
        {
            uint16_t * data = saving_list.back();
            saving_list.pop_back();
            fwrite(data,sizeof(uint16_t),frWidth*dataHeight,file_target); //It is ok if this blocks
            delete[] data;
        }
        else
        {
            //We're waiting for data to get added to the list...
            usleep(250);
        }
    }
    fclose(file_target);
    //We're done!
    printf("done saving!");

}
unsigned int take_object::getFrameHeight()
{
    return frHeight;
}
unsigned int take_object::getFrameWidth()
{
    return frWidth;
}
unsigned int take_object::getDataHeight()
{
    return dataHeight;
}
void take_object::startCapturingDSFMask()
{
    dsfMaskCollected = false;
    dsf->start_mask_collection();
}
void take_object::finishCapturingDSFMask()
{
    dsf->finish_mask_collection();
    dsfMaskCollected = true;
}
/*void take_object::panicSave( std::string raw_file_name )
{
    while(!saving_list.empty());
    boost::thread(&take_object::saveFramesInBuffer,this);
    boost::thread(&take_object::savingLoop,this,raw_file_name);
}*/
void take_object::startSavingRaws(std::string raw_file_name, unsigned int frames_to_save)
{
    save_framenum.store(0,std::memory_order_seq_cst);
#ifdef VERBOSE
    printf("ssr called\n");
#endif
    while(!saving_list.empty())
    {
        //printf("waiting for empty saving list...");
    }
    save_framenum.store(frames_to_save,std::memory_order_seq_cst);
#ifdef VERBOSE
    printf("Begin frame save! @ %s", raw_file_name.c_str());
#endif

    boost::thread(&take_object::savingLoop,this,raw_file_name);
    //setvbuf(raw_save_file,NULL,_IOFBF,frames_to_save*size);

    //do_raw_save = true;
}
void take_object::stopSavingRaws()
{
    save_framenum.store(0,std::memory_order_relaxed);
#ifdef VERBOSE
    printf("stop saving raws!");
#endif
}

void take_object::loadDSFMask(std::string file_name)
{
    //boost::shared_array < float > mask_in(new float[frWidth*frHeight]);
    float * mask_in = new float[frWidth*frHeight];
    FILE * pFile;
    unsigned long size = 0;
    pFile  = fopen(file_name.c_str(), "rb");
    if(pFile==NULL) std::cerr << "error opening raw file" << std::endl;
    else
    {
        fseek (pFile, 0, SEEK_END);   // non-portable
        size=ftell (pFile);
        if(size != (frWidth*frHeight*sizeof(float)))
        {
            std::cerr << "error mask file does not match image size" << std::endl;
            fclose (pFile);
            return;
        }
        rewind(pFile);   // go back to beginning
        fread(mask_in,sizeof(float),frWidth*frHeight,pFile);
        fclose (pFile);
        std::cout << file_name << " read in "<< size << " bytes successfully " <<  std::endl;
    }
    dsf->load_mask(mask_in);
}
void take_object::setStdDev_N(int s)
{
    this->std_dev_filter_N = s;
}
void take_object::changeFFTtype(int t)
{
    whichFFT = t;
}
int take_object::getFFTtype()
{
    return whichFFT;
}
/*
std::vector<float> * take_object::getHistogramBins()
{
    return sdvf->getHistogramBins();
} */
bool take_object::std_dev_ready()
{
    return sdvf->outputReady();
}
void take_object::setInversion( bool checked, unsigned int factor )
{
    inverted = checked;
    invFactor = factor;
}
void take_object::chromaPixRemap( bool checked )
{
    chromaPix = checked;
}
void take_object::update_start_row( int sr )
{ // these will only trigger for vertical mean profiles
    meanStartRow = sr;
}
void take_object::update_end_row( int er )
{
    meanHeight = er;
}
void take_object::updateVertRange( int br, int er )
{ // these will only trigger for horizontal cross profiles
    meanStartRow = br;
    meanHeight = er;
    //std::cout << "meanStartRow: " << meanStartRow << " meanHeight: " << meanHeight << std::endl;
}
void take_object::updateHorizRange( int bc, int ec )
{
    meanStartCol = bc;
    meanWidth = ec;
    //std::cout << "meanStartCol: " << meanStartCol << " meanWidth: " << meanWidth << std::endl;
}
