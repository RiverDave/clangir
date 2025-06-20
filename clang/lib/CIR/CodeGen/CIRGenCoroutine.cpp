//===----- CGCoroutine.cpp - Emit CIR Code for C++ coroutines -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of coroutines.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/ScopeExit.h"

using namespace clang;
using namespace clang::CIRGen;

struct clang::CIRGen::CGCoroData {
  // What is the current await expression kind and how many
  // await/yield expressions were encountered so far.
  // These are used to generate pretty labels for await expressions in LLVM IR.
  cir::AwaitKind CurrentAwaitKind = cir::AwaitKind::Init;

  // Stores the __builtin_coro_id emitted in the function so that we can supply
  // it as the first argument to other builtins.
  cir::CallOp CoroId = nullptr;

  // Stores the result of __builtin_coro_begin call.
  mlir::Value CoroBegin = nullptr;

  // Stores the insertion point for final suspend, this happens after the
  // promise call (return_xxx promise member) but before a cir.br to the return
  // block.
  mlir::Operation *FinalSuspendInsPoint;

  // How many co_return statements are in the coroutine. Used to decide whether
  // we need to add co_return; equivalent at the end of the user authored body.
  unsigned CoreturnCount = 0;

  // The promise type's 'unhandled_exception' handler, if it defines one.
  Stmt *ExceptionHandler = nullptr;
};

// Defining these here allows to keep CGCoroData private to this file.
CIRGenFunction::CGCoroInfo::CGCoroInfo() {}
CIRGenFunction::CGCoroInfo::~CGCoroInfo() {}

static void createCoroData(CIRGenFunction &CGF,
                           CIRGenFunction::CGCoroInfo &CurCoro,
                           cir::CallOp CoroId) {
  if (CurCoro.Data) {
    llvm_unreachable("EmitCoroutineBodyStatement called twice?");

    return;
  }

  CurCoro.Data = std::unique_ptr<CGCoroData>(new CGCoroData);
  CurCoro.Data->CoroId = CoroId;
}

namespace {
// FIXME: both GetParamRef and ParamReferenceReplacerRAII are good template
// candidates to be shared among LLVM / CIR codegen.

// Hunts for the parameter reference in the parameter copy/move declaration.
struct GetParamRef : public StmtVisitor<GetParamRef> {
public:
  DeclRefExpr *Expr = nullptr;
  GetParamRef() {}
  void VisitDeclRefExpr(DeclRefExpr *E) {
    assert(Expr == nullptr && "multilple declref in param move");
    Expr = E;
  }
  void VisitStmt(Stmt *S) {
    for (auto *C : S->children()) {
      if (C)
        Visit(C);
    }
  }
};

// This class replaces references to parameters to their copies by changing
// the addresses in CGF.LocalDeclMap and restoring back the original values in
// its destructor.
struct ParamReferenceReplacerRAII {
  CIRGenFunction::DeclMapTy SavedLocals;
  CIRGenFunction::DeclMapTy &LocalDeclMap;

  ParamReferenceReplacerRAII(CIRGenFunction::DeclMapTy &LocalDeclMap)
      : LocalDeclMap(LocalDeclMap) {}

  void addCopy(DeclStmt const *PM) {
    // Figure out what param it refers to.

    assert(PM->isSingleDecl());
    VarDecl const *VD = static_cast<VarDecl const *>(PM->getSingleDecl());
    Expr const *InitExpr = VD->getInit();
    GetParamRef Visitor;
    Visitor.Visit(const_cast<Expr *>(InitExpr));
    assert(Visitor.Expr);
    DeclRefExpr *DREOrig = Visitor.Expr;
    auto *PD = DREOrig->getDecl();

    auto it = LocalDeclMap.find(PD);
    assert(it != LocalDeclMap.end() && "parameter is not found");
    SavedLocals.insert({PD, it->second});

    auto copyIt = LocalDeclMap.find(VD);
    assert(copyIt != LocalDeclMap.end() && "parameter copy is not found");
    it->second = copyIt->getSecond();
  }

  ~ParamReferenceReplacerRAII() {
    for (auto &&SavedLocal : SavedLocals) {
      LocalDeclMap.insert({SavedLocal.first, SavedLocal.second});
    }
  }
};
} // namespace

// Emit coroutine intrinsic and patch up arguments of the token type.
RValue CIRGenFunction::emitCoroutineIntrinsic(const CallExpr *E,
                                              unsigned int IID) {
  llvm_unreachable("NYI");
}

RValue CIRGenFunction::emitCoroutineFrame() {
  if (CurCoro.Data && CurCoro.Data->CoroBegin) {
    return RValue::get(CurCoro.Data->CoroBegin);
  }
  llvm_unreachable("NYI");
}

static mlir::LogicalResult
emitBodyAndFallthrough(CIRGenFunction &CGF, const CoroutineBodyStmt &S,
                       Stmt *Body,
                       const CIRGenFunction::LexicalScope *currLexScope) {
  if (CGF.emitStmt(Body, /*useCurrentScope=*/true).failed())
    return mlir::failure();
  // Note that LLVM checks CanFallthrough by looking into the availability
  // of the insert block which is kinda brittle and unintuitive, seems to be
  // related with how landing pads are handled.
  //
  // CIRGen handles this by checking pre-existing co_returns in the current
  // scope instead. Are we missing anything?
  //
  // From LLVM IR Gen: const bool CanFallthrough = Builder.GetInsertBlock();
  const bool CanFallthrough = !currLexScope->hasCoreturn();
  if (CanFallthrough)
    if (Stmt *OnFallthrough = S.getFallthroughHandler())
      if (CGF.emitStmt(OnFallthrough, /*useCurrentScope=*/true).failed())
        return mlir::failure();

  return mlir::success();
}

cir::CallOp CIRGenFunction::emitCoroIDBuiltinCall(mlir::Location loc,
                                                  mlir::Value nullPtr) {
  auto int32Ty = builder.getUInt32Ty();

  auto &TI = CGM.getASTContext().getTargetInfo();
  unsigned NewAlign = TI.getNewAlign() / TI.getCharWidth();

  mlir::Operation *builtin = CGM.getGlobalValue(CGM.builtinCoroId);

  cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = CGM.createCIRFunction(
        loc, CGM.builtinCoroId,
        cir::FuncType::get({int32Ty, VoidPtrTy, VoidPtrTy, VoidPtrTy}, int32Ty),
        /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(&getMLIRContext()));
  } else
    fnOp = cast<cir::FuncOp>(builtin);

  return builder.createCallOp(loc, fnOp,
                              mlir::ValueRange{builder.getUInt32(NewAlign, loc),
                                               nullPtr, nullPtr, nullPtr});
}

cir::CallOp CIRGenFunction::emitCoroAllocBuiltinCall(mlir::Location loc) {
  auto boolTy = builder.getBoolTy();
  auto int32Ty = builder.getUInt32Ty();

  mlir::Operation *builtin = CGM.getGlobalValue(CGM.builtinCoroAlloc);

  cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = CGM.createCIRFunction(loc, CGM.builtinCoroAlloc,
                                 cir::FuncType::get({int32Ty}, boolTy),
                                 /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(&getMLIRContext()));
  } else
    fnOp = cast<cir::FuncOp>(builtin);

  return builder.createCallOp(
      loc, fnOp, mlir::ValueRange{CurCoro.Data->CoroId.getResult()});
}

cir::CallOp
CIRGenFunction::emitCoroBeginBuiltinCall(mlir::Location loc,
                                         mlir::Value coroframeAddr) {
  auto int32Ty = builder.getUInt32Ty();
  mlir::Operation *builtin = CGM.getGlobalValue(CGM.builtinCoroBegin);

  cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = CGM.createCIRFunction(
        loc, CGM.builtinCoroBegin,
        cir::FuncType::get({int32Ty, VoidPtrTy}, VoidPtrTy),
        /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(&getMLIRContext()));
  } else
    fnOp = cast<cir::FuncOp>(builtin);

  return builder.createCallOp(
      loc, fnOp,
      mlir::ValueRange{CurCoro.Data->CoroId.getResult(), coroframeAddr});
}

cir::CallOp CIRGenFunction::emitCoroEndBuiltinCall(mlir::Location loc,
                                                   mlir::Value nullPtr) {
  auto boolTy = builder.getBoolTy();
  mlir::Operation *builtin = CGM.getGlobalValue(CGM.builtinCoroEnd);

  cir::FuncOp fnOp;
  if (!builtin) {
    fnOp =
        CGM.createCIRFunction(loc, CGM.builtinCoroEnd,
                              cir::FuncType::get({VoidPtrTy, boolTy}, boolTy),
                              /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(&getMLIRContext()));
  } else
    fnOp = cast<cir::FuncOp>(builtin);

  return builder.createCallOp(
      loc, fnOp, mlir::ValueRange{nullPtr, builder.getBool(false, loc)});
}

mlir::LogicalResult
CIRGenFunction::emitCoroutineBody(const CoroutineBodyStmt &S) {
  auto openCurlyLoc = getLoc(S.getBeginLoc());
  auto nullPtrCst = builder.getNullPtr(VoidPtrTy, openCurlyLoc);

  auto Fn = dyn_cast<cir::FuncOp>(CurFn);
  assert(Fn && "other callables NYI");
  Fn.setCoroutineAttr(mlir::UnitAttr::get(&getMLIRContext()));
  auto coroId = emitCoroIDBuiltinCall(openCurlyLoc, nullPtrCst);
  createCoroData(*this, CurCoro, coroId);

  // Backend is allowed to elide memory allocations, to help it, emit
  // auto mem = coro.alloc() ? 0 : ... allocation code ...;
  auto coroAlloc = emitCoroAllocBuiltinCall(openCurlyLoc);

  // Initialize address of coroutine frame to null
  auto astVoidPtrTy = CGM.getASTContext().VoidPtrTy;
  auto allocaTy = convertTypeForMem(astVoidPtrTy);
  Address coroFrame =
      CreateTempAlloca(allocaTy, getContext().getTypeAlignInChars(astVoidPtrTy),
                       openCurlyLoc, "__coro_frame_addr",
                       /*ArraySize=*/nullptr);

  auto storeAddr = coroFrame.getPointer();
  builder.CIRBaseBuilderTy::createStore(openCurlyLoc, nullPtrCst, storeAddr);
  builder.create<cir::IfOp>(openCurlyLoc, coroAlloc.getResult(),
                            /*withElseRegion=*/false,
                            /*thenBuilder=*/
                            [&](mlir::OpBuilder &b, mlir::Location loc) {
                              builder.CIRBaseBuilderTy::createStore(
                                  loc, emitScalarExpr(S.getAllocate()),
                                  storeAddr);
                              builder.create<cir::YieldOp>(loc);
                            });

  CurCoro.Data->CoroBegin =
      emitCoroBeginBuiltinCall(
          openCurlyLoc,
          builder.create<cir::LoadOp>(openCurlyLoc, allocaTy, storeAddr))
          .getResult();

  // Handle allocation failure if 'ReturnStmtOnAllocFailure' was provided.
  if (S.getReturnStmtOnAllocFailure())
    llvm_unreachable("NYI");

  {
    // FIXME(cir): create a new scope to copy out the params?
    // LLVM create scope cleanups here, but might be due to the use
    // of many basic blocks?
    assert(!cir::MissingFeatures::generateDebugInfo() && "NYI");
    ParamReferenceReplacerRAII ParamReplacer(LocalDeclMap);

    // Create mapping between parameters and copy-params for coroutine
    // function.
    llvm::ArrayRef<const Stmt *> ParamMoves = S.getParamMoves();
    assert((ParamMoves.size() == 0 || (ParamMoves.size() == FnArgs.size())) &&
           "ParamMoves and FnArgs should be the same size for coroutine "
           "function");
    // For zipping the arg map into debug info.
    assert(!cir::MissingFeatures::generateDebugInfo() && "NYI");

    // Create parameter copies. We do it before creating a promise, since an
    // evolution of coroutine TS may allow promise constructor to observe
    // parameter copies.
    for (auto *PM : S.getParamMoves()) {
      if (emitStmt(PM, /*useCurrentScope=*/true).failed())
        return mlir::failure();
      ParamReplacer.addCopy(cast<DeclStmt>(PM));
    }

    if (emitStmt(S.getPromiseDeclStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

    // ReturnValue should be valid as long as the coroutine's return type
    // is not void. The assertion could help us to reduce the check later.
    assert(ReturnValue.isValid() == (bool)S.getReturnStmt());
    // Now we have the promise, initialize the GRO.
    // We need to emit `get_return_object` first. According to:
    // [dcl.fct.def.coroutine]p7
    // The call to get_return_­object is sequenced before the call to
    // initial_suspend and is invoked at most once.
    //
    // So we couldn't emit return value when we emit return statment,
    // otherwise the call to get_return_object wouldn't be in front
    // of initial_suspend.
    if (ReturnValue.isValid()) {
      emitAnyExprToMem(S.getReturnValue(), ReturnValue,
                       S.getReturnValue()->getType().getQualifiers(),
                       /*IsInit*/ true);
    }

    // FIXME(cir): EHStack.pushCleanup<CallCoroEnd>(EHCleanup);
    CurCoro.Data->CurrentAwaitKind = cir::AwaitKind::Init;
    if (emitStmt(S.getInitSuspendStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

    CurCoro.Data->CurrentAwaitKind = cir::AwaitKind::User;

    // FIXME(cir): wrap emitBodyAndFallthrough with try/catch bits.
    if (S.getExceptionHandler())
      assert(!cir::MissingFeatures::unhandledException() && "NYI");
    if (emitBodyAndFallthrough(*this, S, S.getBody(), currLexScope).failed())
      return mlir::failure();

    // Note that LLVM checks CanFallthrough by looking into the availability
    // of the insert block which is kinda brittle and unintuitive, seems to be
    // related with how landing pads are handled.
    //
    // CIRGen handles this by checking pre-existing co_returns in the current
    // scope instead. Are we missing anything?
    //
    // From LLVM IR Gen: const bool CanFallthrough = Builder.GetInsertBlock();
    const bool CanFallthrough = currLexScope->hasCoreturn();
    const bool HasCoreturns = CurCoro.Data->CoreturnCount > 0;
    if (CanFallthrough || HasCoreturns) {
      CurCoro.Data->CurrentAwaitKind = cir::AwaitKind::Final;
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(CurCoro.Data->FinalSuspendInsPoint);
        if (emitStmt(S.getFinalSuspendStmt(), /*useCurrentScope=*/true)
                .failed())
          return mlir::failure();
      }
    }
  }
  return mlir::success();
}

static bool memberCallExpressionCanThrow(const Expr *E) {
  if (const auto *CE = dyn_cast<CXXMemberCallExpr>(E))
    if (const auto *Proto =
            CE->getMethodDecl()->getType()->getAs<FunctionProtoType>())
      if (isNoexceptExceptionSpec(Proto->getExceptionSpecType()) &&
          Proto->canThrow() == CT_Cannot)
        return false;
  return true;
}

// Given a suspend expression which roughly looks like:
//
//   auto && x = CommonExpr();
//   if (!x.await_ready()) {
//      x.await_suspend(...); (*)
//   }
//   x.await_resume();
//
// where the result of the entire expression is the result of x.await_resume()
//
//   (*) If x.await_suspend return type is bool, it allows to veto a suspend:
//      if (x.await_suspend(...))
//        llvm_coro_suspend();
//
// This is more higher level than LLVM codegen, for that one see llvm's
// docs/Coroutines.rst for more details.
namespace {
struct LValueOrRValue {
  LValue LV;
  RValue RV;
};
} // namespace
static LValueOrRValue
emitSuspendExpression(CIRGenFunction &CGF, CGCoroData &Coro,
                      CoroutineSuspendExpr const &S, cir::AwaitKind Kind,
                      AggValueSlot aggSlot, bool ignoreResult,
                      mlir::Block *scopeParentBlock,
                      mlir::Value &tmpResumeRValAddr, bool forLValue) {
  auto *E = S.getCommonExpr();

  auto awaitBuild = mlir::success();
  LValueOrRValue awaitRes;

  auto Binder =
      CIRGenFunction::OpaqueValueMappingData::bind(CGF, S.getOpaqueValue(), E);
  auto UnbindOnExit = llvm::make_scope_exit([&] { Binder.unbind(CGF); });
  auto &builder = CGF.getBuilder();

  [[maybe_unused]] auto awaitOp = builder.create<cir::AwaitOp>(
      CGF.getLoc(S.getSourceRange()), Kind,
      /*readyBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        Expr *condExpr = S.getReadyExpr()->IgnoreParens();
        builder.createCondition(CGF.evaluateExprAsBool(condExpr));
      },
      /*suspendBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        // Note that differently from LLVM codegen we do not emit coro.save
        // and coro.suspend here, that should be done as part of lowering this
        // to LLVM dialect (or some other MLIR dialect)

        // A invalid suspendRet indicates "void returning await_suspend"
        auto suspendRet = CGF.emitScalarExpr(S.getSuspendExpr());

        // Veto suspension if requested by bool returning await_suspend.
        if (suspendRet) {
          // From LLVM codegen:
          // if (SuspendRet != nullptr && SuspendRet->getType()->isIntegerTy(1))
          llvm_unreachable("NYI");
        }

        // Signals the parent that execution flows to next region.
        builder.create<cir::YieldOp>(loc);
      },
      /*resumeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        // Exception handling requires additional IR. If the 'await_resume'
        // function is marked as 'noexcept', we avoid generating this additional
        // IR.
        CXXTryStmt *TryStmt = nullptr;
        if (Coro.ExceptionHandler && Kind == cir::AwaitKind::Init &&
            memberCallExpressionCanThrow(S.getResumeExpr())) {
          llvm_unreachable("NYI");
        }

        // FIXME(cir): the alloca for the resume expr should be placed in the
        // enclosing cir.scope instead.
        if (forLValue)
          awaitRes.LV = CGF.emitLValue(S.getResumeExpr());
        else {
          awaitRes.RV =
              CGF.emitAnyExpr(S.getResumeExpr(), aggSlot, ignoreResult);
          if (!awaitRes.RV.isIgnored()) {
            // Create the alloca in the block before the scope wrapping
            // cir.await.
            tmpResumeRValAddr = CGF.emitAlloca(
                "__coawait_resume_rval", awaitRes.RV.getScalarVal().getType(),
                loc, CharUnits::One(),
                builder.getBestAllocaInsertPoint(scopeParentBlock));
            // Store the rvalue so we can reload it before the promise call.
            builder.CIRBaseBuilderTy::createStore(
                loc, awaitRes.RV.getScalarVal(), tmpResumeRValAddr);
          }
        }

        if (TryStmt) {
          llvm_unreachable("NYI");
        }

        // Returns control back to parent.
        builder.create<cir::YieldOp>(loc);
      });

  assert(awaitBuild.succeeded() && "Should know how to codegen");
  return awaitRes;
}

static RValue emitSuspendExpr(CIRGenFunction &CGF,
                              const CoroutineSuspendExpr &E,
                              cir::AwaitKind kind, AggValueSlot aggSlot,
                              bool ignoreResult) {
  RValue rval;
  auto scopeLoc = CGF.getLoc(E.getSourceRange());

  // Since we model suspend / resume as an inner region, we must store
  // resume scalar results in a tmp alloca, and load it after we build the
  // suspend expression. An alternative way to do this would be to make
  // every region return a value when promise.return_value() is used, but
  // it's a bit awkward given that resume is the only region that actually
  // returns a value.
  mlir::Block *currEntryBlock = CGF.currLexScope->getEntryBlock();
  [[maybe_unused]] mlir::Value tmpResumeRValAddr;

  // No need to explicitly wrap this into a scope since the AST already uses a
  // ExprWithCleanups, which will wrap this into a cir.scope anyways.
  rval = emitSuspendExpression(CGF, *CGF.CurCoro.Data, E, kind, aggSlot,
                               ignoreResult, currEntryBlock, tmpResumeRValAddr,
                               /*forLValue*/ false)
             .RV;

  if (ignoreResult || rval.isIgnored())
    return rval;

  if (rval.isScalar()) {
    rval = RValue::get(CGF.getBuilder().create<cir::LoadOp>(
        scopeLoc, rval.getScalarVal().getType(), tmpResumeRValAddr));
  } else if (rval.isAggregate()) {
    // This is probably already handled via AggSlot, remove this assertion
    // once we have a testcase and prove all pieces work.
    llvm_unreachable("NYI");
  } else { // complex
    llvm_unreachable("NYI");
  }
  return rval;
}

RValue CIRGenFunction::emitCoawaitExpr(const CoawaitExpr &E,
                                       AggValueSlot aggSlot,
                                       bool ignoreResult) {
  return emitSuspendExpr(*this, E, CurCoro.Data->CurrentAwaitKind, aggSlot,
                         ignoreResult);
}

RValue CIRGenFunction::emitCoyieldExpr(const CoyieldExpr &E,
                                       AggValueSlot aggSlot,
                                       bool ignoreResult) {
  return emitSuspendExpr(*this, E, cir::AwaitKind::Yield, aggSlot,
                         ignoreResult);
}

mlir::LogicalResult CIRGenFunction::emitCoreturnStmt(CoreturnStmt const &S) {
  ++CurCoro.Data->CoreturnCount;
  currLexScope->setCoreturn();

  const Expr *RV = S.getOperand();
  if (RV && RV->getType()->isVoidType() && !isa<InitListExpr>(RV)) {
    // Make sure to evaluate the non initlist expression of a co_return
    // with a void expression for side effects.
    // FIXME(cir): add scope
    // RunCleanupsScope cleanupScope(*this);
    emitIgnoredExpr(RV);
  }
  if (emitStmt(S.getPromiseCall(), /*useCurrentScope=*/true).failed())
    return mlir::failure();
  // Create a new return block (if not existent) and add a branch to
  // it. The actual return instruction is only inserted during current
  // scope cleanup handling.
  auto loc = getLoc(S.getSourceRange());
  auto *retBlock = currLexScope->getOrCreateRetBlock(*this, loc);
  CurCoro.Data->FinalSuspendInsPoint = builder.create<cir::BrOp>(loc, retBlock);

  // Insert the new block to continue codegen after branch to ret block,
  // this will likely be an empty block.
  builder.createBlock(builder.getBlock()->getParent());

  // TODO(cir): LLVM codegen for a cleanup on cleanupScope here.
  return mlir::success();
}
