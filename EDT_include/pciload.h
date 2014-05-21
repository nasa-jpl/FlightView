/* #pragma ident "@(#)pciload.h	1.9 07/11/01 EDT" */
#ifndef INCLUDE_pciload_h
#define INCLUDE_pciload_h

#ifdef _NT_
 #define DIRECTORY_CHAR '\\'
#else
 #define DIRECTORY_CHAR '/'
#endif

#define	MAX_STRING	128

#define MIN_BIT_SIZE_ANY	0x4000
#define MIN_BIT_SIZE_X		0x8000
#define MIN_BIT_SIZE_XLA	0x14000
#define MIN_BIT_SIZE_BT		0x20000
#define MIN_BIT_SIZE_SPI	0x60000
#define MIN_BIT_SIZE_AMD512	0x300000

#define XTYPE_X		0
#define XTYPE_BT	1
#define XTYPE_LTX	2
#define XTYPE_SPI	3
#define XTYPE_BT2	4

#define	EDT_ROM_JUMPER	0x01
#define	EDT_5_VOLT	0x02
/*
 * status bits
 * bits 2-7 are Xilinx ID for 4028 and above.
 * shift down 2 and we get a number
 * 0x00 - 4028XLA boot controller
 * 0x01 - XC2S150 boot controller 
 * 0x02 - XC2S200 boot controller without protect 29LV040B FPROM 
 * 0x03 - XC2S200 boot controller with protect restored (future 8Mbit FPROM)
 * 0x04 - XC2S100 boot controller
 * 0x05 - XC3S1200E boot controller
 * 0x06 - XC5VLX30T boot controller
 * 0x07 - XC5VLX50T boot controller
 * 0x08 - XC5VLX70T boot controller
 * 0x15 - EP2SGX30D boot controller (ALERT: for sure? ask jg)
 */
#define STAT_PROTECTED	0x01
#define STAT_5V		0x02
/* #define STAT_IDMASK	0xfc */
#define STAT_IDMASK	0x7c /* change jsc 12/6/02 to include xc2s100 */
#define STAT_IDSHFT	2
#define STAT_XC4028XLA	0x00
#define STAT_LTX_XC2S300E 0x00
#define STAT_XC2S150	0x01
#define STAT_XC2S200_NP	0x02
#define STAT_XC2S200	0x03
#define STAT_XC2S100    0x04
#define STAT_XC3S1200E  0x05
#define STAT_XC5VLX30T  0x06
#define STAT_XC5VLX50T  0x07
#define STAT_XC5VLX70T  0x08
#define STAT_XC5VLX30T_A 0x09
#define STAT_XC6SLX45   0x0a
#define STAT_EP2SGX30D  0x15

/*
 * command bits for the 4028xla
 * boot controller
 */
#define	BT_READ		0x0100
#define	BT_WRITE	0x0200
#define BT_INC_ADD	0x0400
#define	BT_A0		0x0800
#define	BT_A1		0x1000
#define	BT_RSVD 	0x2000
#define	BT_REINIT	0x4000
#define BT_EN_READ	0x8000
#define	BT_LD_LO	BT_WRITE
#define	BT_LD_MID	BT_WRITE | BT_A0
#define	BT_LD_HI	BT_WRITE | BT_A1
#define	BT_LD_ROM	BT_WRITE | BT_A0 | BT_A1
#define BT_RD_ROM	BT_READ | BT_A0 | BT_A1 | BT_EN_READ
#define BT_RD_FSTAT	BT_READ | BT_EN_READ
#define BT_RD_PALVER	BT_READ | BT_A0 | BT_EN_READ
#define	BT_MASK		0xff00

#define IS_DEFAULT_SECTOR -6167


extern int sect;

#define MAX_BOARD_SEARCH 8
#define NUM_DEVICE_TYPES 5

typedef enum {
    UnknownMagic,
    XilinxMagic = 1,
    AlteraMagic = 2
} FPGAMagic;


#define BAD_PARMS_BLOCK ((EdtPromParmBlock *) 0xffffffff)

#ifndef _KERNEL

#include "edt_bitload.h"

EDTAPI void warnuser(EdtDev *edt_p, char *fname, int sector);
EDTAPI void warnuser_ltx(EdtDev *edt_p, char *fname, int unit, int hub);

EDTAPI int check_id_stuff(char *bid, char *pid, int devid, int verify_only, char *fname);
EDTAPI void getinfo(EdtDev *edt_p, int promcode, int segment, char *pid, char *esn, char *osn, int verbose);
EDTAPI void getinfonf(EdtDev *edt_p, int promcode, int segment, char *pid, char *esn, char *osn, int verbose);

EDTAPI void program_sns(EdtDev *edt_p, int promtype, char *new_esn, char *new_osn, int sector, char *id, int verbose);
EDTAPI void print_flashstatus(char stat, int sector, int frdata, int verbose);
EDTAPI void edt_get_sns(EdtDev *edt_p, char *esn, char *osn);
EDTAPI void edt_get_osn(EdtDev *edt_p, char *osn);
EDTAPI void edt_get_esn(EdtDev *edt_p, char *esn);
EDTAPI int ask_reboot(EdtDev *edt_p);
EDTAPI void ask_options(char *options);
EDTAPI void ask_rev(int *rev);
EDTAPI void ask_clock(int *clock, char *extra_txt);
EDTAPI void ask_pn(char *pn);
EDTAPI void ask_sn(char *sn);
EDTAPI int ask_addinfo();
EDTAPI void strip_newline(char *s);
EDTAPI int pciload_isdigit_str(char *s);


typedef int (*EdtPciLoadVerify) (EdtDev *edt_p, 
                                 EdtBitfile *bitfile, 
                                 EdtPromData *pdata, 
                                 int promcode, 
                                 int segment, 
                                 int vfonly, 
                                 int warn, 
                                 int verbose);

int
program_verify_4013xla(EdtDev *edt_p, 
                        EdtBitfile *bitfile, 
                        EdtPromData *pdata, 
                        int promcode, 
                        int segment, 
                        int vfonly, 
                        int warn,
                        int verbose);
int
program_verify_XC2S300E(EdtDev *edt_p, 
                        EdtBitfile *bitfile, 
                        EdtPromData *pdata, 
                        int promcode, 
                        int segment, 
                        int vfonly, 
                        int warn,
                        int verbose);
int
program_verify_XC2S200(EdtDev *edt_p, 
                        EdtBitfile *bitfile, 
                        EdtPromData *pdata, 
                        int promcode, 
                        int segment, 
                        int vfonly, 
                        int warn,
                        int verbose);
int
program_verify_XC2S150(EdtDev *edt_p, 
                        EdtBitfile *bitfile, 
                        EdtPromData *pdata, 
                        int promcode, 
                        int segment, 
                        int vfonly, 
                        int warn,
                        int verbose);
int
program_verify_4028XLA(EdtDev *edt_p, 
                        EdtBitfile *bitfile, 
                        EdtPromData *pdata, 
                        int promcode, 
                        int segment, 
                        int vfonly, 
                        int warn,
                        int verbose);
int
program_verify_default(EdtDev *edt_p, 
                        EdtBitfile *bitfile, 
                        EdtPromData *pdata, 
                        int promcode, 
                        int segment, 
                        int vfonly, 
                        int warn,
                        int verbose);

int program_verify_SPI(EdtDev *edt_p, 
                       EdtBitfile *bitfile, 
                       EdtPromData *pdata, 
                       int promcode, 
                       int sector, 
                       int verify_only, 
                       int warn, 
                       int verbose);

int program_verify_4013E(EdtDev *edt_p, 
                       EdtBitfile *bitfile, 
                       EdtPromData *pdata, 
                       int promcode, 
                       int sector, 
                       int verify_only, 
                       int warn, 
                       int verbose);

int program_verify_4013XLA(EdtDev *edt_p, 
                       EdtBitfile *bitfile, 
                       EdtPromData *pdata, 
                       int promcode, 
                       int sector, 
                       int verify_only, 
                       int warn, 
                       int verbose);

#endif /* ! KERNEL */

#endif /* INCLUDE_pciload_h */
