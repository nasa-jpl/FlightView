
#ifndef _INITCAM_H
#define _INITCAM_H

#include "edtinc.h"

/*
 * struct used by readcfg.c and initcam.c, for stuff
 * other than what's in depdendent
 */
typedef struct {
	u_int startdma ;
	u_int enddma ;
	u_int flushdma ;
	u_int timeout ;
} Edtinfo;

EDTAPI int  pdv_readcfg(const char *cfgfile, Dependent *pm, Edtinfo *ei) ;
EDTAPI int  pdv_readcfg_emb(char *cfgfile, Dependent *pm, Edtinfo *ei) ;
EDTAPI void pdv_dep_set_default(PdvDependent * dd_p);
EDTAPI int  printcfg(Dependent *pm) ;


EDTAPI int   pdv_initcam(EdtDev *edt_p, Dependent *dd_p, int unit, Edtinfo *edtinfo,
				const char *cfgfname, char *bitdir, int pdv_debug);
EDTAPI int   pdv_initcam_readcfg(char *cfgfile, Dependent * dd_p, Edtinfo * ei_p);
EDTAPI int   pdv_initcam_checkfoi(EdtDev *edt_p, Edtinfo *p_edtinfo, int unit);
EDTAPI Dependent *pdv_alloc_dependent(void);
EDTAPI int   pdv_initcam_load_bitfile(EdtDev * edt_p, Dependent * dd_p,
				int unit, char *bitdir, const char *cfgfname);
EDTAPI int   pdv_initcam_load_bitfile(EdtDev * edt_p, Dependent * dd_p,
				int unit, char *bitdir, const char *cfgfname);
EDTAPI int   pdv_initcam_kbs_check_and_reset_camera(EdtDev * edt_p, Dependent * dd_p);
EDTAPI int   pdv_initcam_reset_camera(EdtDev *edt_p, Dependent *dd_p, Edtinfo *p_edtinfo);
EDTAPI void  pdv_initcam_set_rci(EdtDev *edt_p, int rci_unit) ;

#endif /* _INITCAM_H */
