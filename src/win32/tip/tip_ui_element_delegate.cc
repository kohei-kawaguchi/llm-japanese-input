// Copyright 2010-2021, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "win32/tip/tip_ui_element_delegate.h"

#include <atlbase.h>
#include <atlstr.h>
#include <msctf.h>
#include <wil/com.h>
#include <wil/resource.h>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "base/win32/com.h"
#include "base/win32/wide_char.h"
#include "protocol/candidate_window.pb.h"
#include "protocol/commands.pb.h"
#include "protocol/renderer_command.pb.h"
#include "win32/tip/tip_dll_module.h"
#include "win32/tip/tip_edit_session.h"
#include "win32/tip/tip_private_context.h"
#include "win32/tip/tip_resource.h"
#include "win32/tip/tip_text_service.h"

namespace mozc {
namespace win32 {
namespace tsf {
namespace {

using ::ATL::CStringW;
using ::mozc::commands::CandidateList;
using ::mozc::commands::Output;
using ::mozc::commands::Status;
using IndicatorInfo = ::mozc::commands::RendererCommand_IndicatorInfo;

constexpr size_t kPageSize = 9;

// This GUID is used in Windows Vista/7/8 by MS-IME to represents if the
// candidate window is visible or not.
// TODO(yukawa): Make sure if it is safe to use this GUID.
// {B7A578D2-9332-438A-A403-4057D05C3958}
constexpr GUID kGuidCUASCandidateMessageCompartment = {
    0xb7a578d2,
    0x9332,
    0x438a,
    {0xa4, 0x03, 0x40, 0x57, 0xd0, 0x5c, 0x39, 0x58}};

#ifdef GOOGLE_JAPANESE_INPUT_BUILD

// {8F51B5E5-5CF9-45D8-83B3-53CE203354C2}
constexpr GUID KGuidNonobservableSuggestWindow = {
    0x8f51b5e5,
    0x5cf9,
    0x45d8,
    {0x83, 0xb3, 0x53, 0xce, 0x20, 0x33, 0x54, 0xc2}};

// {3D53878A-8596-4689-B50D-3338D52B2EFB}
constexpr GUID KGuidObservableSuggestWindow = {
    0x3d53878a,
    0x8596,
    0x4689,
    {0xb5, 0xd, 0x33, 0x38, 0xd5, 0x2b, 0x2e, 0xfb}};

// {FED897F2-940C-40F1-B149-A931E03FB821}
constexpr GUID KGuidCandidateWindow = {
    0xfed897f2,
    0x940c,
    0x40f1,
    {0xb1, 0x49, 0xa9, 0x31, 0xe0, 0x3f, 0xb8, 0x21}};

// {170F6CC4-913D-4FF9-9DEA-432D08DCB0FF}
constexpr GUID KGuidIndicatorWindow = {
    0x170f6cc4,
    0x913d,
    0x4ff9,
    {0x9d, 0xea, 0x43, 0x2d, 0x8, 0xdc, 0xb0, 0xff}};

#else  // GOOGLE_JAPANESE_INPUT_BUILD

// {09E1BC34-BD7B-4B85-B67A-8E6F3C0082E7}
constexpr GUID KGuidNonobservableSuggestWindow = {
    0x09e1bc34,
    0xbd7b,
    0x4b85,
    {0xb6, 0x7a, 0x8e, 0x6f, 0x3c, 0x00, 0x82, 0xe7}};

// {C442F346-ED39-4A88-89C4-B09AAA56C9D8}
constexpr GUID KGuidObservableSuggestWindow = {
    0xc442f346,
    0xed39,
    0x4a88,
    {0x89, 0xc4, 0xb0, 0x9a, 0xaa, 0x56, 0xc9, 0xd8}};

// {E69291F5-DDB7-4DA3-91AA-4B101B2519CC}
constexpr GUID KGuidCandidateWindow = {
    0xe69291f5,
    0xddb7,
    0x4da3,
    {0x91, 0xaa, 0x4b, 0x10, 0x1b, 0x25, 0x19, 0xcc}};

// {52E28D45-4B27-4CC3-9FE4-B516572496D0}
constexpr GUID KGuidIndicatorWindow = {
    0x52e28d45,
    0x4b27,
    0x4cc3,
    {0x9f, 0xe4, 0xb5, 0x16, 0x57, 0x24, 0x96, 0xd0}};

#endif  // GOOGLE_JAPANESE_INPUT_BUILD

wil::unique_bstr GetResourceString(UINT resource_id) {
  CStringW str;
  str.LoadStringW(TipDllModule::module_handle(), resource_id);
  return MakeUniqueBSTR(std::wstring_view(str.GetBuffer(), str.GetLength()));
}

class TipUiElementDelegateImpl final : public TipUiElementDelegate {
 public:
  TipUiElementDelegateImpl(const TipUiElementDelegateImpl&) = delete;
  TipUiElementDelegateImpl& operator=(const TipUiElementDelegateImpl&) = delete;
  TipUiElementDelegateImpl(wil::com_ptr_nothrow<TipTextService> text_service,
                           wil::com_ptr_nothrow<ITfContext> context,
                           TipUiElementDelegateFactory::ElementType type)
      : text_service_(std::move(text_service)),
        context_(std::move(context)),
        type_(type) {}

 private:
  bool IsObservable() const override {
    switch (type_) {
      case TipUiElementDelegateFactory::kConventionalObservableSuggestWindow:
      case TipUiElementDelegateFactory::kConventionalCandidateWindow:
        return true;
      default:
        return false;
    }
  }

  // The ITfUIElement interface methods
  HRESULT GetDescription(BSTR* description) override {
    if (description == nullptr) {
      return E_INVALIDARG;
    }
    *description = nullptr;
    switch (type_) {
      case TipUiElementDelegateFactory::kConventionalUnobservableSuggestWindow:
        *description =
            GetResourceString(IDS_UNOBSERVABLE_SUGGEST_WINDOW).release();
        return S_OK;
      case TipUiElementDelegateFactory::kConventionalObservableSuggestWindow:
        *description =
            GetResourceString(IDS_OBSERVABLE_SUGGEST_WINDOW).release();
        return S_OK;
      case TipUiElementDelegateFactory::kConventionalCandidateWindow:
        *description = GetResourceString(IDS_CANDIDATE_WINDOW).release();
        return S_OK;
      case TipUiElementDelegateFactory::kConventionalIndicatorWindow:
        *description = GetResourceString(IDS_INDICATOR_WINDOW).release();
        return S_OK;
      default:
        return E_UNEXPECTED;
    }
  }

  HRESULT GetGUID(GUID* guid) override {
    if (guid == nullptr) {
      return E_INVALIDARG;
    }
    switch (type_) {
      case TipUiElementDelegateFactory::kConventionalUnobservableSuggestWindow:
        *guid = KGuidNonobservableSuggestWindow;
        return S_OK;
      case TipUiElementDelegateFactory::kConventionalObservableSuggestWindow:
        *guid = KGuidObservableSuggestWindow;
        return S_OK;
      case TipUiElementDelegateFactory::kConventionalCandidateWindow:
        *guid = KGuidCandidateWindow;
        return S_OK;
      case TipUiElementDelegateFactory::kConventionalIndicatorWindow:
        *guid = KGuidIndicatorWindow;
        return S_OK;
      default:
        *guid = GUID_NULL;
        return E_UNEXPECTED;
    }
  }

  HRESULT Show(BOOL show) override {
    const bool old_shown = shown_;
    shown_ = !!show;
    if (old_shown != shown_ && IsObservable()) {
      auto compartment_mgr = ComQuery<ITfCompartmentMgr>(context_);
      if (compartment_mgr) {
        // Update a hidden compartment to generate
        // IMN_OPENCANDIDATE/IMN_CLOSECANDIDATE notifications for the
        // application compatibility.
        wil::com_ptr_nothrow<ITfCompartment> compartment;
        if (SUCCEEDED(compartment_mgr->GetCompartment(
                kGuidCUASCandidateMessageCompartment, &compartment))) {
          wil::unique_variant var;
          var.vt = VT_I4;
          var.lVal = shown_ ? TRUE : FALSE;
          compartment->SetValue(text_service_->GetClientID(), var.addressof());
        }
      }
      // TODO(yukawa): Update UI.
    }
    return S_OK;
  }

  HRESULT IsShown(BOOL* show) override {
    if (show == nullptr) {
      return E_INVALIDARG;
    }
    *show = (shown_ ? TRUE : FALSE);
    return S_OK;
  }

  // The ITfCandidateListUIElement interface methods
  HRESULT GetUpdatedFlags(DWORD* flags) override {
    DCHECK(IsCandidateWindowLike());

    if (flags == nullptr) {
      return E_INVALIDARG;
    }
    *flags = 0;
    // If TF_CLUIE_STRING is included into |flags|, TSF calls back
    // ITfCandidateListUIElement::GetString for all the candidates,
    // which might be a huge bottleneck. So do not include this flag
    // whenever possible.
    if (TestModifiedAndUpdateLastCandidate()) {
      *flags |= (TF_CLUIE_STRING | TF_CLUIE_COUNT);
    }
    *flags |= (TF_CLUIE_SELECTION | TF_CLUIE_CURRENTPAGE | TF_CLUIE_PAGEINDEX);
    return S_OK;
  }

  HRESULT GetDocumentMgr(ITfDocumentMgr** document_manager) override {
    DCHECK(IsCandidateWindowLike());

    if (document_manager == nullptr) {
      return E_INVALIDARG;
    }
    return context_->GetDocumentMgr(document_manager);
  }

  HRESULT GetCount(UINT* count) override {
    DCHECK(IsCandidateWindowLike());

    if (count == nullptr) {
      return E_INVALIDARG;
    }
    *count = 0;
    TipPrivateContext* private_context =
        text_service_->GetPrivateContext(context_.get());
    if (private_context == nullptr) {
      return E_FAIL;
    }
    const Output& output = private_context->last_output();
    if (!output.has_all_candidate_words()) {
      return S_OK;
    }
    *count = output.all_candidate_words().candidates_size();
    return S_OK;
  }

  HRESULT GetSelection(UINT* index) override {
    DCHECK(IsCandidateWindowLike());

    if (index == nullptr) {
      return E_INVALIDARG;
    }
    *index = 0;
    TipPrivateContext* private_context =
        text_service_->GetPrivateContext(context_.get());
    if (private_context == nullptr) {
      return E_FAIL;
    }
    const Output& output = private_context->last_output();
    if (!output.has_all_candidate_words()) {
      return S_OK;
    }
    *index = output.all_candidate_words().focused_index();
    return S_OK;
  }

  HRESULT GetString(UINT index, BSTR* text) override {
    DCHECK(IsCandidateWindowLike());

    if (text == nullptr) {
      return E_INVALIDARG;
    }
    *text = nullptr;
    TipPrivateContext* private_context =
        text_service_->GetPrivateContext(context_.get());
    if (private_context == nullptr) {
      return E_FAIL;
    }
    const Output& output = private_context->last_output();
    if (!output.has_all_candidate_words()) {
      return E_FAIL;
    }
    const CandidateList& list = output.all_candidate_words();
    // Convert |index| to the index within output_->candidates().
    const int visible_index = index;
    if (visible_index >= list.candidates_size()) {
      return E_FAIL;
    }
    std::wstring wide_text = Utf8ToWide(list.candidates(visible_index).value());
    *text = MakeUniqueBSTR(wide_text).release();
    return S_OK;
  }

  HRESULT GetPageIndex(UINT* index, UINT size, UINT* page_count) override {
    DCHECK(IsCandidateWindowLike());

    if (page_count == nullptr) {
      return E_INVALIDARG;
    }
    TipPrivateContext* private_context =
        text_service_->GetPrivateContext(context_.get());
    if (private_context == nullptr) {
      return E_FAIL;
    }
    const Output& output = private_context->last_output();
    if (!output.has_all_candidate_words()) {
      return E_FAIL;
    }
    const CandidateList& list = output.all_candidate_words();
    const size_t max_page = (list.candidates_size() / kPageSize);
    *page_count = max_page + 1;

    if (index == nullptr) {
      // An application can pass nullptr as |index| to obtain only page_count.
      return S_OK;
    }

    if (size < *page_count) {
      return E_NOT_SUFFICIENT_BUFFER;
    }
    for (size_t i = 0; i < *page_count; ++i) {
      index[i] = i * kPageSize;
    }
    return S_OK;
  }

  HRESULT SetPageIndex(UINT* index, UINT page_count) override {
    DCHECK(IsCandidateWindowLike());

    return E_NOTIMPL;
  }

  HRESULT GetCurrentPage(UINT* current_page) override {
    DCHECK(IsCandidateWindowLike());

    if (current_page == nullptr) {
      return E_INVALIDARG;
    }
    *current_page = 0;
    TipPrivateContext* private_context =
        text_service_->GetPrivateContext(context_.get());
    if (private_context == nullptr) {
      return E_FAIL;
    }
    const Output& output = private_context->last_output();
    if (!output.has_all_candidate_words()) {
      return S_OK;
    }
    *current_page = output.all_candidate_words().focused_index() / kPageSize;
    return S_OK;
  }

  // The ITfCandidateListUIElementBehavior interface methods
  HRESULT SetSelection(UINT index) override {
    DCHECK(IsCandidateWindowLike());

    TipPrivateContext* private_context =
        text_service_->GetPrivateContext(context_.get());
    if (private_context == nullptr) {
      return E_FAIL;
    }
    const Output& output = private_context->last_output();
    if (!output.has_all_candidate_words()) {
      return E_FAIL;
    }
    const CandidateList& list = output.all_candidate_words();
    if (list.candidates_size() <= index) {
      return E_INVALIDARG;
    }
    const int id = list.candidates(index).id();
    if (!TipEditSession::SelectCandidateAsync(text_service_.get(),
                                              context_.get(), id)) {
      return E_FAIL;
    }
    return S_OK;
  }

  HRESULT Finalize() override {
    DCHECK(IsCandidateWindowLike());

    if (!TipEditSession::SubmitAsync(text_service_.get(), context_.get())) {
      return E_FAIL;
    }
    return S_OK;
  }

  HRESULT Abort() override {
    DCHECK(IsCandidateWindowLike());

    // Currently equals to Finalize().
    if (!TipEditSession::SubmitAsync(text_service_.get(), context_.get())) {
      return E_FAIL;
    }
    return S_OK;
  }

  HRESULT GetString(BSTR* text) override {
    DCHECK(IsIndicator());

    if (text == nullptr) {
      return E_INVALIDARG;
    }

    TipPrivateContext* private_context =
        text_service_->GetPrivateContext(context_.get());
    if (private_context == nullptr) {
      *text = MakeUniqueBSTR(L"").release();
      return S_OK;
    }
    if (!private_context->last_output().has_status()) {
      *text = MakeUniqueBSTR(L"").release();
      return S_OK;
    }
    const Status& status = private_context->last_output().status();
    if (status.has_activated() && !status.activated()) {
      *text = MakeUniqueBSTR(L"A").release();
      return S_OK;
    }
    if (!status.has_mode()) {
      *text = MakeUniqueBSTR(L"").release();
      return S_OK;
    }

    std::wstring msg;
    switch (status.mode()) {
      case commands::DIRECT:
        DLOG(FATAL) << "Unexpected last output status mode: DIRECT";
        break;
      case commands::HIRAGANA:
        msg = L"\u3042";
        break;
      case commands::FULL_KATAKANA:
        msg = L"\u30AB";
        break;
      case commands::HALF_ASCII:
        msg = L"_A";
        break;
      case commands::FULL_ASCII:
        msg = L"\uFF21";
        break;
      case commands::HALF_KATAKANA:
        msg = L"_\uFF76";
        break;
      default:
        break;
    }

    *text = MakeUniqueBSTR(msg).release();
    return S_OK;
  }

  // Returns true if the candidate list is updated. When this function returns
  // false, you need not update the list of candidate strings at this time.
  // Note that this function updates |last_candidate_list_| internally.
  bool TestModifiedAndUpdateLastCandidate() {
    TipPrivateContext* private_context =
        text_service_->GetPrivateContext(context_.get());
    if (private_context == nullptr) {
      return true;
    }
    const Output& output = private_context->last_output();
    if (!output.has_all_candidate_words()) {
      return true;
    }
    const CandidateList& list = output.all_candidate_words();
    if (last_candidate_list_.candidates_size() != list.candidates_size()) {
      last_candidate_list_ = list;
      return true;
    }
    for (int i = 0; i < list.candidates_size(); ++i) {
      if (last_candidate_list_.candidates(i).value() !=
          list.candidates(i).value()) {
        last_candidate_list_ = list;
        return true;
      }
    }
    return false;
  }

  bool IsCandidateWindowLike() const {
    switch (type_) {
      case TipUiElementDelegateFactory::kConventionalUnobservableSuggestWindow:
        return true;
      case TipUiElementDelegateFactory::kConventionalObservableSuggestWindow:
        return true;
      case TipUiElementDelegateFactory::kConventionalCandidateWindow:
        return true;
      case TipUiElementDelegateFactory::kConventionalIndicatorWindow:
        return false;
      default:
        return false;
    }
  }

  bool IsIndicator() const {
    switch (type_) {
      case TipUiElementDelegateFactory::kConventionalUnobservableSuggestWindow:
        return false;
      case TipUiElementDelegateFactory::kConventionalObservableSuggestWindow:
        return false;
      case TipUiElementDelegateFactory::kConventionalCandidateWindow:
        return false;
      case TipUiElementDelegateFactory::kConventionalIndicatorWindow:
        return true;
      default:
        return false;
    }
  }

  wil::com_ptr_nothrow<TipTextService> text_service_;
  wil::com_ptr_nothrow<ITfContext> context_;
  const TipUiElementDelegateFactory::ElementType type_;
  CandidateList last_candidate_list_;
  bool shown_;
};

}  // namespace

std::unique_ptr<TipUiElementDelegate> TipUiElementDelegateFactory::Create(
    wil::com_ptr_nothrow<TipTextService> text_service,
    wil::com_ptr_nothrow<ITfContext> context, ElementType type) {
  return std::make_unique<TipUiElementDelegateImpl>(std::move(text_service),
                                                    std::move(context), type);
}

}  // namespace tsf
}  // namespace win32
}  // namespace mozc
