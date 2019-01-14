/** \file
 * \brief ILU preconditioners
 */

#ifndef STRUCT3D_ILU_H
#define STRUCT3D_ILU_H

#include "s3d_sgspreconditioners.hpp"

class StrILU_preconditioner : public SGS_like_preconditioner
{
public:
	StrILU_preconditioner(const SMat& lhs, const PreconParams params);
	void updateOperator();
};

#endif
