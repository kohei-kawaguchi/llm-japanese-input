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

#include "win32/tip/tip_lang_bar.h"

#include <ctfutb.h>
#include <guiddef.h>
#include <wil/com.h>
#include <windows.h>

#include <cstdint>
#include <iterator>
#include <utility>

#include "absl/log/log.h"
#include "base/win32/com.h"
#include "base/win32/hresultor.h"
#include "protocol/commands.pb.h"
#include "win32/tip/tip_dll_module.h"
#include "win32/tip/tip_lang_bar_callback.h"
#include "win32/tip/tip_lang_bar_menu.h"
#include "win32/tip/tip_resource.h"

namespace mozc {
namespace win32 {
namespace tsf {
namespace {

// The GUID of the help menu in the system language bar.

// {ED9D5450-EBE6-4255-8289-F8A31E687228}
constexpr GUID kSystemLangBarHelpMenu = {
    0xED9D5450,
    0xEBE6,
    0x4255,
    {0x82, 0x89, 0xF8, 0xA3, 0x1E, 0x68, 0x72, 0x28}};

#ifdef GOOGLE_JAPANESE_INPUT_BUILD

// {D8C8D5EB-8213-47CE-95B7-BA3F67757F94}
constexpr GUID kTipLangBarItem_Button = {
    0xd8c8d5eb,
    0x8213,
    0x47ce,
    {0x95, 0xb7, 0xba, 0x3f, 0x67, 0x75, 0x7f, 0x94}};

// {0EAB48C4-F798-4CC8-91FA-087B24F520A8}
constexpr GUID kTipLangBarItem_ToolButton = {
    0xeab48c4, 0xf798, 0x4cc8, {0x91, 0xfa, 0x8, 0x7b, 0x24, 0xf5, 0x20, 0xa8}};

// {6D46F0F2-2924-4666-9B89-4F23699B2203}
constexpr GUID kTipLangBarItem_HelpMenu = {
    0x6d46f0f2,
    0x2924,
    0x4666,
    {0x9b, 0x89, 0x4f, 0x23, 0x69, 0x9b, 0x22, 0x3}};

#else  // GOOGLE_JAPANESE_INPUT_BUILD

// {840B1FE0-25A1-4837-97FB-E000A1CF3C94}
constexpr GUID kTipLangBarItem_Button = {
    0x840b1fe0,
    0x25a1,
    0x4837,
    {0x97, 0xfb, 0xe0, 0x00, 0xa1, 0xcf, 0x3c, 0x94}};

// {6F646990-067B-4535-BDE4-BB9C8A978873}
constexpr GUID kTipLangBarItem_ToolButton = {
    0x6f646990,
    0x067b,
    0x4535,
    {0xbd, 0xe4, 0xbb, 0x9c, 0x8a, 0x97, 0x88, 0x73}};

// {E99488C3-EEB1-4CA3-B972-4FFF410E83DA}
constexpr GUID kTipLangBarItem_HelpMenu = {
    0xe99488c3,
    0xeeb1,
    0x4ca3,
    {0xb9, 0x72, 0x4f, 0xff, 0x41, 0x0e, 0x83, 0xda}};

#endif  // GOOGLE_JAPANESE_INPUT_BUILD

constexpr bool kShowInTaskbar = true;

TipLangBarCallback::ItemId GetItemId(DWORD composition_mode) {
  switch (composition_mode) {
    case commands::DIRECT:
      return TipLangBarCallback::kDirect;
    case commands::HIRAGANA:
      return TipLangBarCallback::kHiragana;
    case commands::FULL_KATAKANA:
      return TipLangBarCallback::kFullKatakana;
    case commands::HALF_ASCII:
      return TipLangBarCallback::kHalfAlphanumeric;
    case commands::FULL_ASCII:
      return TipLangBarCallback::kFullAlphanumeric;
    case commands::HALF_KATAKANA:
      return TipLangBarCallback::kHalfKatakana;
    default:
      LOG(ERROR) << "Unknown composition mode: " << composition_mode;
      return TipLangBarCallback::kDirect;
  }
}

}  // namespace

// Initializes button menus in the language bar.
HRESULT TipLangBar::InitLangBar(TipLangBarCallback* text_service) {
  HRESULT result = S_OK;

  // TODO(yukawa): Optimize this method. We do not need to obtain an instance of
  // ITfLangBarItemMgr unless there remains something to be initialized for
  // LangBar.

  // A workaround to satisfy both b/6106437 and b/6641460.
  // Keep the instance in |lang_bar_item_mgr_|.
  if (!lang_bar_item_mgr_) {
    result = TF_CreateLangBarItemMgr(lang_bar_item_mgr_.put());
    if (FAILED(result)) {
      return result;
    }
    if (!lang_bar_item_mgr_) {
      return E_FAIL;
    }
  }

  const TipLangBarMenuItem kInputMenuDisabled = {kTipLangBarItemTypeDefault, 0,
                                                 IDS_DISABLED, IDI_DISABLED_NT,
                                                 IDI_DISABLED};

  if (input_button_menu_ == nullptr) {
    // Add the "Input Mode" button.
    constexpr TipLangBarMenuItem kInputMenu[] = {
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kHiragana,
         IDS_HIRAGANA, IDI_HIRAGANA_NT, IDI_HIRAGANA},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kFullKatakana,
         IDS_FULL_KATAKANA, IDI_FULL_KATAKANA_NT, IDI_FULL_KATAKANA},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kFullAlphanumeric,
         IDS_FULL_ALPHANUMERIC, IDI_FULL_ALPHANUMERIC_NT,
         IDI_FULL_ALPHANUMERIC},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kHalfKatakana,
         IDS_HALF_KATAKANA, IDI_HALF_KATAKANA_NT, IDI_HALF_KATAKANA},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kHalfAlphanumeric,
         IDS_HALF_ALPHANUMERIC, IDI_HALF_ALPHANUMERIC_NT,
         IDI_HALF_ALPHANUMERIC},
        {kTipLangBarItemTypeRadioChecked, TipLangBarCallback::kDirect,
         IDS_DIRECT, IDI_DIRECT_NT, IDI_DIRECT},
        {kTipLangBarItemTypeSeparator, 0, 0, 0, 0},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kCancel, IDS_CANCEL, 0,
         0},
    };

    constexpr bool kMenuButton = true;
    auto input_button_menu = MakeComPtr<TipLangBarToggleButton>(
        text_service, kTipLangBarItem_Button, kMenuButton, kShowInTaskbar);
    if (input_button_menu == nullptr) {
      return E_OUTOFMEMORY;
    }

    result = input_button_menu->Init(TipDllModule::module_handle(),
                                     IDS_INPUTMODE, &kInputMenu[0],
                                     std::size(kInputMenu), kInputMenuDisabled);
    if (result != S_OK) {
      return result;
    }
    lang_bar_item_mgr_->AddItem(input_button_menu.get());
    input_button_menu_ = std::move(input_button_menu);
  }

  if (!input_mode_button_for_win8_) {
    // Add the "Input Mode" button.
    constexpr TipLangBarMenuItem kInputMenu[] = {
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kHiragana,
         IDS_HIRAGANA, IDI_HIRAGANA_NT, IDI_HIRAGANA},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kFullKatakana,
         IDS_FULL_KATAKANA, IDI_FULL_KATAKANA_NT, IDI_FULL_KATAKANA},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kFullAlphanumeric,
         IDS_FULL_ALPHANUMERIC, IDI_FULL_ALPHANUMERIC_NT,
         IDI_FULL_ALPHANUMERIC},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kHalfKatakana,
         IDS_HALF_KATAKANA, IDI_HALF_KATAKANA_NT, IDI_HALF_KATAKANA},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kHalfAlphanumeric,
         IDS_HALF_ALPHANUMERIC, IDI_HALF_ALPHANUMERIC_NT,
         IDI_HALF_ALPHANUMERIC},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kDirect, IDS_DIRECT,
         IDI_DIRECT_NT, IDI_DIRECT},
        {kTipLangBarItemTypeSeparator, 0, 0, 0, 0},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kDictionary,
         IDS_DICTIONARY, IDI_DICTIONARY_NT, IDI_DICTIONARY},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kWordRegister,
         IDS_WORD_REGISTER, IDI_DICTIONARY_NT, IDI_DICTIONARY},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kProperty,
         IDS_PROPERTY, IDI_PROPERTY_NT, IDI_PROPERTY},
        {kTipLangBarItemTypeSeparator, 0, 0, 0, 0},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kAbout, IDS_ABOUT, 0,
         0},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kHelp, IDS_HELP, 0, 0},
    };

    constexpr bool kNonMenuButton = false;
    auto input_mode_menu = MakeComPtr<TipLangBarToggleButton>(
        text_service, GUID_LBI_INPUTMODE, kNonMenuButton, kShowInTaskbar);
    if (input_mode_menu == nullptr) {
      return E_OUTOFMEMORY;
    }

    result = input_mode_menu->Init(TipDllModule::module_handle(),
                                   IDS_WIN8_TRAY_ITEM, kInputMenu,
                                   std::size(kInputMenu), kInputMenuDisabled);
    if (FAILED(result)) {
      return result;
    }
    result = lang_bar_item_mgr_->AddItem(input_mode_menu.get());
    input_mode_button_for_win8_ = std::move(input_mode_menu);
  }

  if (tool_button_menu_ == nullptr) {
    // Add the "Tool" button.
    // TODO(yukawa): Make an Icon for kWordRegister.
    const TipLangBarMenuItem kToolMenu[] = {
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kDictionary,
         IDS_DICTIONARY, IDI_DICTIONARY_NT, IDI_DICTIONARY},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kWordRegister,
         IDS_WORD_REGISTER, IDI_DICTIONARY_NT,
         IDI_DICTIONARY},  // Use Dictionary icon temporarily
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kProperty,
         IDS_PROPERTY, IDI_PROPERTY_NT, IDI_PROPERTY},
        {kTipLangBarItemTypeSeparator, 0, 0, 0, 0},
        {kTipLangBarItemTypeDefault, TipLangBarCallback::kCancel, IDS_CANCEL, 0,
         0},
    };

    // Always show the tool icon so that a user can find the icon.
    // This setting is different from that of MS-IME but we believe this is
    // more friendly. See b/2275683
    auto tool_button = MakeComPtr<TipLangBarMenuButton>(
        text_service, kTipLangBarItem_ToolButton, kShowInTaskbar);
    if (tool_button == nullptr) {
      return E_OUTOFMEMORY;
    }

    result = tool_button->Init(TipDllModule::module_handle(), IDS_TOOL,
                               &kToolMenu[0], std::size(kToolMenu), IDI_TOOL_NT,
                               IDI_TOOL);
    if (result != S_OK) {
      return result;
    }
    lang_bar_item_mgr_->AddItem(tool_button.get());
    tool_button_menu_ = std::move(tool_button);
  }

  if (help_menu_ == nullptr) {
    // Add the "Help" items to the system language bar help menu.
    const TipLangBarMenuItem kHelpMenu[] = {
        {0, TipLangBarCallback::kAbout, IDS_ABOUT, 0, 0},
        {0, TipLangBarCallback::kHelp, IDS_HELP, 0, 0},
    };

    auto help_menu = MakeComPtr<TipSystemLangBarMenu>(text_service,
                                                      kTipLangBarItem_HelpMenu);
    if (help_menu == nullptr) {
      return E_OUTOFMEMORY;
    }

    result = help_menu->Init(TipDllModule::module_handle(), &kHelpMenu[0],
                             std::size(kHelpMenu));
    if (result != S_OK) {
      return result;
    }

    wil::com_ptr_nothrow<ITfLangBarItem> help_menu_item;
    result =
        lang_bar_item_mgr_->GetItem(kSystemLangBarHelpMenu, &help_menu_item);
    if (result != S_OK) {
      return result;
    }
    ASSIGN_OR_RETURN_HRESULT(auto source,
                             ComQueryHR<ITfSource>(help_menu_item));
    result = source->AdviseSink(IID_ITfSystemLangBarItemSink, help_menu.get(),
                                &help_menu_cookie_);
    if (result != S_OK) {
      return result;
    }

    help_menu_ = std::move(help_menu);
  }

  return result;
}

// IMPORTANT: See b/6106437 and b/6641460 before you change this method.
HRESULT TipLangBar::UninitLangBar() {
  HRESULT result = S_OK;

  // A workaround to satisfy both b/6106437 and b/6641460.
  // Retrieve the instance from |lang_bar_item_mgr_|.
  wil::com_ptr_nothrow<ITfLangBarItemMgr> item = std::move(lang_bar_item_mgr_);
  if (!item) {
    return E_FAIL;
  }

  if (input_mode_button_for_win8_) {
    item->RemoveItem(input_mode_button_for_win8_.get());
    input_mode_button_for_win8_.reset();
  }
  if (input_button_menu_) {
    item->RemoveItem(input_button_menu_.get());
    input_button_menu_.reset();
  }
  if (tool_button_menu_) {
    item->RemoveItem(tool_button_menu_.get());
    tool_button_menu_.reset();
  }

  if (help_menu_ && (help_menu_cookie_ != TF_INVALID_COOKIE)) {
    wil::com_ptr_nothrow<ITfLangBarItem> help_menu_item;
    result = item->GetItem(kSystemLangBarHelpMenu, &help_menu_item);
    if (result == S_OK) {
      auto source = ComQuery<ITfSource>(help_menu_item);
      if (source) {
        result = source->UnadviseSink(help_menu_cookie_);
        if (result == S_OK) {
          help_menu_cookie_ = TF_INVALID_COOKIE;
          help_menu_.reset();
        }
      }
    }
  }

  return result;
}

HRESULT TipLangBar::UpdateMenu(bool enabled, uint32_t composition_mode) {
  const UINT menu_id = GetItemId(composition_mode);
  input_button_menu_->SelectMenuItem(menu_id);
  input_mode_button_for_win8_->SelectMenuItem(menu_id);
  input_button_menu_->SetEnabled(enabled);
  tool_button_menu_->SetEnabled(enabled);
  input_mode_button_for_win8_->SetEnabled(enabled);
  return S_OK;
}

bool TipLangBar::IsInitialized() const {
  return input_button_menu_ || input_mode_button_for_win8_;
}

}  // namespace tsf
}  // namespace win32
}  // namespace mozc
