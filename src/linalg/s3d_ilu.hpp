/** \file
 * \brief ILU preconditioners
 */

#ifndef STRUCT3D_ILU_H
#define STRUCT3D_ILU_H

#include "s3d_sgspreconditioners.hpp"
#include "s3d_isai.hpp"

void async_ilu0(const SMat& lhs, const PreconParams params, s3d::vector<sreal>& dinv);

class StrILU_preconditioner : public SGS_like_preconditioner
{
public:
	StrILU_preconditioner(const SMat& lhs, const PreconParams params);
	void updateOperator();
};

/// Asynchronous ILU(0) applied by ISAI iterations
class AILU_ISAI_preconditioner : public ISAI_preconditioner
{
public:
	AILU_ISAI_preconditioner(const SMat& lhs, const PreconParams params);
	void updateOperator();
};

#endif
