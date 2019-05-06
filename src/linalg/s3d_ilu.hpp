/** \file
 * \brief ILU preconditioners
 */

#ifndef STRUCT3D_ILU_H
#define STRUCT3D_ILU_H

#include "s3d_sgspreconditioners.hpp"

void async_ilu0(const SMat& lhs, const PreconParams params, s3d::vector<sreal>& dinv);

class StrILU_preconditioner : public SGS_like_preconditioner
{
public:
	StrILU_preconditioner(const SMat& lhs, const PreconParams params);
	void updateOperator();

protected:
	void updateOperatorWithSeparateLoops();
};

#endif
