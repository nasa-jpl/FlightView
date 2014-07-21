#include "frame_c.hpp"


frame_c::frame_c() {
	delete_counter = 5;
	async_filtering_done = 0;
	valid_std_dev = 0;
}
int8_t frame_c::get_delete_counter()
{
	boost::shared_lock<boost::shared_mutex> shlock(dc_mux);
	return delete_counter;
}
void frame_c::set_delete_counter(int8_t val)
{
	boost::unique_lock<boost::shared_mutex> ulock(dc_mux);
	delete_counter = val;
}
int8_t frame_c::get_async_filtering_done()
{
	boost::shared_lock<boost::shared_mutex> shlock(af_mux);
	return async_filtering_done;
}
void frame_c::set_async_filtering_done(int8_t val)
{
	boost::unique_lock<boost::shared_mutex> ulock(af_mux);
	async_filtering_done = val;
}
int8_t frame_c::get_valid_std_dev()
{
	boost::shared_lock<boost::shared_mutex> shlock(vsd_mux);
	return valid_std_dev;
}
void frame_c::set_valid_std_dev(int8_t val)
{
	boost::unique_lock<boost::shared_mutex> ulock(vsd_mux);
	valid_std_dev = val;
}
