/**
 * @file
 * Camera Link Simulator Library header file, for use with PCI DV CLS and PCIe DV
 * C-Link with Simulator FPGA loaded.
 */
#ifndef _CLSIM_H_
#define _CLSIM_H_

#ifdef __cplusplus

extern "C" {

#endif

/**
 * @defgroup cls EDT Camera Link Simulator Library
 * The Camera Link Simulator (CLS) Library provids routines and registers
 * specific to the CLS board.
 *
 * The source code for the library is in clsim_lib.c and clsim_lib.h.
 *
 *
 *
 */

#define PDV_CLS_DEFAULT_HGAP	300
#define PDV_CLS_DEFAULT_VGAP	400
#define PDV_CLS_DEFAULT_FREQ    20.0

/**
 * @addtogroup cls
 * @{
 */

EDTAPI void pdv_cls_dump_state(PdvDev *pdv_p);
EDTAPI void pdv_cls_dump_geometry(PdvDev *pdv_p);
EDTAPI int pdv_cls_set_dep(PdvDev *pdv_p);
EDTAPI int pdv_cls_dep_sanity_check(PdvDev *pdv_p);
EDTAPI void pdv_cls_set_size(PdvDev *pdv_p,
		int taps,
		int depth ,
		int width,
		int height,
		int hblank,
		int totalwidth,
		int vblank,
		int totalheight);

EDTAPI void pdv_cls_set_line_timing(PdvDev *pdv_p,
		int width,
		int taps,
		int Hfvstart,
		int Hfvend,
		int Hlvstart,
		int Hlvend,
		int Hrvstart,
		int Hrvend);

EDTAPI void pdv_cls_set_linescan(PdvDev *pdv_p, int state);
EDTAPI void pdv_cls_set_lvcont(PdvDev *pdv_p, int state);
EDTAPI void pdv_cls_set_rven(PdvDev *pdv_p, int state);
EDTAPI void pdv_cls_set_uartloop(PdvDev *pdv_p, int state);
EDTAPI void pdv_cls_set_smallok(PdvDev *pdv_p, int state);
EDTAPI void pdv_cls_set_intlven(PdvDev *pdv_p, int state);
EDTAPI void pdv_cls_set_firstfc(PdvDev *pdv_p, int state);
EDTAPI void pdv_cls_set_datacnt(PdvDev *pdv_p, int state);
EDTAPI void pdv_cls_set_led(PdvDev *pdv_p, int state);
EDTAPI void pdv_cls_set_trigsrc(PdvDev *pdv_p, int state);
EDTAPI void pdv_cls_set_trigpol(PdvDev *pdv_p, int state);
EDTAPI void pdv_cls_set_trigframe(PdvDev *pdv_p, int state);
EDTAPI void pdv_cls_set_trigline(PdvDev *pdv_p, int state);
EDTAPI void pdv_cls_sim_start(PdvDev *pdv_p);
EDTAPI void pdv_cls_sim_stop(PdvDev *pdv_p);
EDTAPI void pdv_cls_init_serial(PdvDev *pdv_p);
EDTAPI void pdv_cls_set_height(PdvDev *pdv_p, int rasterlines, int vblank);
EDTAPI void pdv_cls_set_width(PdvDev *pdv_p, int width, int hblank);
EDTAPI void pdv_cls_set_clock(EdtDev *edt_p, double freq) ;
EDTAPI void pe8dvcls_set_clock(EdtDev *pdv_p, double freq) ;
EDTAPI void pe8dvcls_set_clock_nominal(EdtDev *pdv_p, double freq, double nominal) ;
EDTAPI void pdv_cls_set_fill(PdvDev *pdv_p, u_char left, u_char right);
EDTAPI void pdv_cls_set_readvalid(PdvDev *pdv_p, 
		u_short HrvStart, u_short HrvEnd);
EDTAPI void pdv_cls_set_rven(PdvDev *pdv_p, int enable);
EDTAPI void pdv_cls_set_intlven(PdvDev *pdv_p, int enable);
EDTAPI void pdv_cls_set_led(PdvDev *pdv_p, int power_state);

EDTAPI void pdv_cls_setup_interleave(PdvDev *pdv_p,
		short tap0start, short tap0delta, 
		short tap1start, short tap1delta,
		short tap2start, short tap2delta,
		short tap3start, short tap3delta);

EDTAPI int pdv_cls_get_vgap(PdvDev *pdv_p);
EDTAPI int pdv_cls_get_hgap(PdvDev *pdv_p);

/* These are used by clsiminit if no value for blanking is specified */
#define PDV_CLS_DEFAULT_HGAP	300
#define PDV_CLS_DEFAULT_VGAP	400

EDTAPI double pdv_cls_frame_time(PdvDev *pdv_p);

/** @} */ /* end cls group */

#ifdef __cplusplus

}

#endif

#endif /* _CLSIM_H_ */

