#include <stdlib.h>
#include <stdio.h>
#include "edtinc.h"
#include "constant_filter.cuh"

//Really 14, but 16 will work.
#define BYTES_PER_PIXEL 2

static void usage(char *progname, char *errmsg);
void parse_args(int argc, char **argv);
char   *progname ;
char edt_devname[128];
char    bmpfname[128] = "test.bmp";
char    bmpfname_cu[128] = "test_filterd.bmp";

char    rawfname[128] = "test.raw";
char    rawfname_cu[128] = "test_filterd.raw";


int     channel = 0;
int     loops = 1;
int     numbufs = 4;
int      unit = 0;

int main(int argc, char **argv)
{
	int     i;
	int     overrun, overruns=0;
	int     timeouts, last_timeouts = 0;
	int     recovering_timeout = FALSE;
	char   *cameratype;
	int     started;
	u_char *image_p;
	PdvDev *pdv_p;
	char    errstr[64];
	int     width, height, depth;


	if (edt_devname[0])
	{
		unit = edt_parse_unit_channel(edt_devname, edt_devname, EDT_INTERFACE, &channel);
	}
	else
	{
		strcpy(edt_devname, EDT_INTERFACE);
	}

	if ((pdv_p = pdv_open_channel(edt_devname, unit, channel)) == NULL)
	{
		sprintf(errstr, "pdv_open_channel(%s%d_%d)", edt_devname, unit, channel);
		pdv_perror(errstr);
		return (1);
	}

	/*
	 * get image size and name for display, save, printfs, etc.
	 */
	width = pdv_get_width(pdv_p);
	height = pdv_get_height(pdv_p);
	depth = pdv_get_depth(pdv_p);
	cameratype = pdv_get_cameratype(pdv_p);

	/*
	 * allocate four buffers for optimal pdv ring buffer pipeline (reduce if
	 * memory is at a premium)
	 */
	pdv_multibuf(pdv_p, numbufs);

	printf("reading %d image%s from '%s'\nwidth %d height %d depth %d\n",
			loops, loops == 1 ? "" : "s", cameratype, width, height, depth);

	/*
	 * prestart the first image or images outside the loop to get the
	 * pipeline going. Start multiple images unless force_single set in
	 * config file, since some cameras (e.g. ones that need a gap between
	 * images or that take a serial command to start every image) don't
	 * tolerate queueing of multiple images
	 */
	if (pdv_p->dd_p->force_single)
	{
		pdv_start_image(pdv_p);
		started = 1;
	}
	else
	{
		pdv_start_images(pdv_p, numbufs);
		started = numbufs;
	}

	for (i = 0; i < loops; i++)
	{
		/*
		 * get the image and immediately start the next one (if not the last
		 * time through the loop). Processing (saving to a file in this case)
		 * can then occur in parallel with the next acquisition
		 */
		// printf("image %d\r", i + 1);
		// fflush(stdout);
		image_p = pdv_wait_image(pdv_p);

			if ((overrun = (edt_reg_read(pdv_p, PDV_STAT) & PDV_OVERRUN)))
			    ++overruns;

		if (i < loops - started) // If we have started fewer than 4 loops
		{
			pdv_start_image(pdv_p);
		}
		 timeouts = pdv_timeouts(pdv_p);

		/*
		 * check for timeouts or data overruns -- timeouts occur when data
		 * is lost, camera isn't hooked up, etc, and application programs
		 * should always check for them. data overruns usually occur as a
		 * result of a timeout but should be checked for separately since
		 * ROI can sometimes mask timeouts
		 */
		if (timeouts > last_timeouts)
		 {

		 /* pdv_timeout_cleanup helps recover gracefully after a timeout,
		 * particularly if multiple buffers were prestarted
		 */
		 pdv_timeout_restart(pdv_p, TRUE);
		 last_timeouts = timeouts;
		 recovering_timeout = TRUE;
		 printf("\ntimeout....\n");
			} else if (recovering_timeout)
			{
			    pdv_timeout_restart(pdv_p, TRUE);
			    recovering_timeout = FALSE;
			    printf("\nrestarted....\n");
			}
		char * image_p_filterd = apply_constant_filter(image_p, width, height, 20000);
		if (*bmpfname)
		{	printf("writing bmp to %s\n", bmpfname);
			dvu_write_bmp(bmpfname, image_p, width, height);
			dvu_write_bmp(bmpfname_cu, image_p_filterd, width, height);

		}
		if (*rawfname)
		{	printf("writing raw to %s\n", rawfname);
			dvu_write_raw(height*width*2, image_p, rawfname);
			dvu_write_raw(height*width*2, image_p_filterd, rawfname_cu);

		}

	}
	puts("");

	printf("%d images %d timeouts %d overruns\n", loops, last_timeouts, overruns);
	printf("sizeof uchar: %i", sizeof(u_char));
	/*
	 * if we got timeouts it indicates there is a problem
	 */
	if (last_timeouts)
		printf("check camera and connections\n");
	pdv_close(pdv_p);

	exit(0);
	parse_args(argc, argv);


	printf("goodbye");
	return 0;
}

void parse_args(int argc, char **argv)
{
	progname = argv[0];


	edt_devname[0] = '\0';
	*bmpfname = '\0';

	/*
	 * process command line arguments
	 */
	--argc;
	++argv;
	while (argc && ((argv[0][0] == '-') || (argv[0][0] == '/')))
	{
		switch (argv[0][1])
		{
		case 'u':		/* device unit number */
			++argv;
			--argc;
			if (argc < 1)
				usage(progname, "Error: option 'u' requires an argument\n");
			if  ((argv[0][0] >= '0') && (argv[0][0] <= '9'))
				unit = atoi(argv[0]);
			else strncpy(edt_devname, argv[0], sizeof(edt_devname) - 1);
			break;

		case 'c':		/* device channel number */
			++argv;
			--argc;
			if (argc < 1)
			{
				usage(progname, "Error: option 'c' requires a numeric argument\n");
			}
			if ((argv[0][0] >= '0') && (argv[0][0] <= '9'))
			{
				channel = atoi(argv[0]);
			}
			else
			{
				usage(progname, "Error: option 'c' requires a numeric argument\n");
			}
			break;

		case 'N':
			++argv;
			--argc;
			if (argc < 1)
			{
				usage(progname, "Error: option 'N' requires a numeric argument\n");
			}
			if ((argv[0][0] >= '0') && (argv[0][0] <= '9'))
			{
				numbufs = atoi(argv[0]);
			}
			else
			{
				usage(progname, "Error: option 'N' requires a numeric argument\n");
			}
			break;

		case 'b':		/* bitmap save filename */
			++argv;
			--argc;
			strcpy(bmpfname, argv[0]);
			break;

		case 'l':
			++argv;
			--argc;
			if (argc < 1)
			{
				usage(progname, "Error: option 'l' requires a numeric argument\n");
			}
			if ((argv[0][0] >= '0') && (argv[0][0] <= '9'))
			{
				loops = atoi(argv[0]);
			}
			else
			{
				usage(progname, "Error: option 'l' requires a numeric argument\n");
			}
			break;

		case '-':
			if (strcmp(argv[0], "--help") == 0) {
				usage(progname, "");
				exit(0);
			} else {
				fprintf(stderr, "unknown option: %s\n", argv[0]);
				usage(progname, "");
				exit(1);
			}
			break;


		default:
			fprintf(stderr, "unknown flag -'%c'\n", argv[0][1]);
		case '?':
		case 'h':
			usage(progname, "");
			exit(0);
		}
		argc--;
		argv++;
	}
}

static void
usage(char *progname, char *errmsg)
{
	puts(errmsg);
	printf("%s: simple example program that acquires images from an\n", progname);
	printf("EDT Digital Video Interface board (PCI DV, PCI DVK, etc.)\n");
	puts("");
	printf("usage: %s [-b fname] [-l loops] [-N numbufs] [-u unit] [-c channel]\n", progname);
#ifdef _NT_
	printf("  -b fname        output to MS bitmap file\n");
#else
	printf("  -b fname        output to Sun Raster file\n");
#endif
	printf("  -l loops        number of loops (images to take)\n");
	printf("  -N numbufs      number of ring buffers (see users guide) (default 4)\n");
	printf("  -u unit         %s unit number (default 0)\n", EDT_INTERFACE);
	printf("  -c channel      %s channel number (default 0)\n", EDT_INTERFACE);
	printf("  -h              this help message\n");
	exit(1);
}
