"""Apply checked, reproducible integration patches to the pinned upstream source."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import subprocess

root = Path(sys.argv[1]).resolve()
env = root / "vendor/ygo-agent"
core = root / "vendor/ygopro-core-current"
scripts = root / "vendor/ygopro-scripts-current"
database = root / "vendor/ygopro-database-current/locales/en-US/cards.cdb"
header = env / "ygoenv/ygoenv/ygopro/ygopro.h"
source = subprocess.check_output(["git", "-C", str(env), "show", "HEAD:ygoenv/ygoenv/ygopro/ygopro.h"], text=True)


def replace(old, new):
    global source
    if old not in source:
        raise RuntimeError(f"Upstream patch context changed: {old[:100]}")
    source = source.replace(old, new)


replace('namespace ygopro {', '''using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using int32 = int32_t;
using int64 = int64_t;
using byte = uint8_t;
namespace ygopro {
static std::vector<std::string> integration_errors;
static std::map<std::string, int> integration_prompt_counts;
static std::string last_integration_response;
static uint32_t integration_log(intptr_t duel, uint32_t type) {
  char buffer[4096]{};
  get_log_message(duel, buffer);
  integration_errors.emplace_back(buffer);
  return 0;
}
''')
replace('card.set_setcode(setcode);', 'write_setcode(card.setcode, setcode);')
# End Phase remains a legal choice when Main Phase 2 is also available.
replace('''      if (to_ep) {
        if (!to_m2) {
          legal_actions_.push_back(
            LegalAction::phase(ActionPhase::End));
          int cmd_idx = legal_actions_.size();
          if (verbose_) {
            pl->notify(fmt::format("{}: End phase.", cmd_idx));
          }
        }
      }''', '''      if (to_ep) {
        legal_actions_.push_back(LegalAction::phase(ActionPhase::End));
      }''')
replace('''    msg_ = int(data_[dp_++]);''', '''    msg_ = int(data_[dp_++]);
    integration_prompt_counts[msg_to_string(msg_)]++;''')
replace('''    } else if (msg_ == MSG_CONFIRM_CARDS) {
      auto player = read_u8();
      auto size = read_u8();''', '''    } else if (msg_ == MSG_CONFIRM_CARDS) {
      auto player = read_u8();
      read_u8(); // current core's skip-panel flag
      auto size = read_u8();''')
replace('''    set_responsei(pduel, value);''', '''    last_integration_response = fmt::format("message {}, response {}, request {}", msg_, value, fmt::join(std::vector<uint8_t>(data_, data_ + dl_), ","));
    set_responsei(pduel, value);''')
replace('''    set_responseb(pduel, buf);''', '''    last_integration_response = fmt::format("message {}, response bytes {}, request {}", msg_, fmt::join(std::vector<uint8_t>(buf, buf + std::min(int(buf[0]) + 1, 32)), ","), fmt::join(std::vector<uint8_t>(data_, data_ + dl_), ","));
    set_responseb(pduel, buf);''')
replace('throw std::runtime_error("Retry");', '''throw std::runtime_error(fmt::format("Rejected response: {}; current bytes {}, dp {}", last_integration_response, fmt::join(std::vector<uint8_t>(data_, data_ + dl_), ","), dp_));''')
replace('uint8_t read_u8() { return data_[dp_++]; }', '''uint8_t read_u8() {
    if (dp_ >= dl_) throw std::runtime_error("Protocol read overflow (u8)");
    return data_[dp_++];
  }''')
replace('''    uint16_t v = *reinterpret_cast<uint16_t *>(data_ + dp_);''', '''    if (dp_ + 2 > dl_) throw std::runtime_error("Protocol read overflow (u16)");
    uint16_t v = *reinterpret_cast<uint16_t *>(data_ + dp_);''')
replace('''    uint32 v = *reinterpret_cast<uint32_t *>(data_ + dp_);''', '''    if (dp_ + 4 > dl_) throw std::runtime_error("Protocol read overflow (u32)");
    uint32 v = *reinterpret_cast<uint32_t *>(data_ + dp_);''')
replace('''      auto spe_count = read_u8();
      bool forced = read_u8();''', '''      auto spe_count = read_u8();
      bool forced = false;''')
replace('''        auto flag = read_u8();
        CardCode code = read_u32();''', '''        auto flag = read_u8();
        forced = bool(read_u8()) || forced;
        CardCode code = read_u32();''')
replace('''fmt::format("Unknown message {}, length {}, dp {}",
        msg_to_string(msg_), dl_, dp_)''', '''fmt::format("Unknown message {} (id {}), length {}, dp {}, bytes {}",
        msg_to_string(msg_), msg_, dl_, dp_, fmt::join(std::vector<uint8_t>(data_, data_ + dl_), ","))''')
replace('''    std::mt19937 rnd(seed);
    // return create_duel(rnd());
    duel* pduel = new duel();
    pduel->random.reset(rnd());
    return (intptr_t)pduel;''', '    return create_duel(seed);')
replace('''    // end_duel(pduel);
    duel* pd = (duel*)pduel;
    delete pd;''', '    end_duel(pduel);')
# Core-generated Lua includes must be read lazily, not limited to three helpers.
replace('''    fmt::println("[script_reader_callback] Script not found: " + path);
    throw std::runtime_error("[script_reader_callback] Script not found: " + path);''', '''    int len = 0;
    byte* buffer = read_card_script(path, &len);
    if (!buffer && path != "./script/c0.lua")
      integration_errors.emplace_back("Missing script: " + path);
    cards_script_[path] = {buffer, len};
    it = cards_script_.find(path);''')
replace('  set_card_reader(card_reader_callback);', '  set_message_handler(integration_log);\n  set_card_reader(card_reader_callback);')
# Upstream loses card IDs when filling visible board observations.
replace('_set_obs_card_(f_cards, offset, c, hide);', '_set_obs_card_(f_cards, offset, c, hide, card_id);')
# Expose exact error conditions. Python refuses these transitions.
replace('"info:num_options"_.Bind', '"info:integration_error"_.Bind(Spec<int>({}, {0, 1})),\n        "info:action_overflow"_.Bind(Spec<int>({}, {0, 1})),\n        "info:num_options"_.Bind')
replace('''    int n_options = legal_actions_.size();
    state["reward"_] = reward;''', '''    int n_options = legal_actions_.size();
    state["info:integration_error"_] = int(!integration_errors.empty());
    state["info:action_overflow"_] = int(n_options > max_options());
    state["reward"_] = reward;''')
# Timeout must never look like a game win.
replace('''    state["reward"_] = 1.0;
    state["info:to_play"_] = 1;''', '''    state["reward"_] = 0.0;
    state["info:integration_error"_] = 1;
    state["info:action_overflow"_] = 0;
    state["info:to_play"_] = 1;''')
# Stop hiding the end-phase discard decision from the learner.
start = source.index('      if (discard_hand_) {', source.index('} else if (msg_ == MSG_SELECT_CARD)'))
end = source.index('      // TODO(1):', start)
source = source[:start] + '      discard_hand_ = false;\n\n' + source[end:]
# Optional selection is legal, including a zero-card response.
replace('''      if (min == 0) {
        throw std::runtime_error("Min == 0 not implemented for select card");
      }''', '')
replace('''      init_multi_select(min, max, 0, specs);

      to_play_ = player;''', '''      init_multi_select(min, max, 0, specs);
      if (min == 0) legal_actions_.push_back(LegalAction::finish());
      if (cancelable) legal_actions_.push_back(LegalAction::cancel());

      to_play_ = player;''')
replace('''  void _callback_multi_select(int idx, bool finish) {
    const auto &action = legal_actions_[idx];''', '''  void _callback_multi_select(int idx, bool finish) {
    const auto &action = legal_actions_[idx];
    if (action.act_ == ActionAct::Cancel) {
      ms_idx_ = -1;
      YGO_SetResponsei(pduel_, -1);
      return;
    }''')
# Selection/unselection is reversible; upstream silently omitted unselect actions.
replace('''      // unselect not allowed (no regrets)
      dp_ += 8 * unselect_size;''', '''      for (int j = 0; j < unselect_size; ++j) {
        dp_ += 4;
        auto controller = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        auto pos = read_u8();
        select_specs.push_back(ls_to_spec(loc, seq, pos, controller != player));
      }''')
replace('''      if (finishable) {
        legal_actions_.push_back(LegalAction::finish());
      }''', '''      if (finishable || cancelable) {
        legal_actions_.push_back(LegalAction::finish());
      }''')
replace('''    MSG_ANNOUNCE_CARD,
};''', '''    MSG_ANNOUNCE_CARD, MSG_ANNOUNCE_RACE,
};''')
replace('''    } else if (msg == MSG_ANNOUNCE_ATTRIB) {
      _set_obs_action_attrib(feat, i, action.attribute_);''', '''    } else if (msg == MSG_ANNOUNCE_ATTRIB || msg == MSG_ANNOUNCE_RACE) {
      for (int byte = 0; byte < 4; ++byte)
        feat(i, 8 + byte) = uint8_t(action.response_ >> (8 * byte));''')
start = source.index('    } else if (msg_ == MSG_ANNOUNCE_ATTRIB) {')
end = source.index('    } else if (msg_ == MSG_ANNOUNCE_CARD) {', start)
source = source[:start] + '''    } else if (msg_ == MSG_ANNOUNCE_ATTRIB || msg_ == MSG_ANNOUNCE_RACE) {
      auto player = read_u8();
      int count = read_u8();
      auto available = read_u32();
      std::vector<int> bits;
      for (int i = 0; i < 32; ++i)
        if (available & (uint32_t(1) << i)) bits.push_back(i);
      std::function<void(int, int, uint32_t)> enumerate;
      enumerate = [&](int begin, int remaining, uint32_t mask) {
        if (legal_actions_.size() > max_options()) return;
        if (!remaining) {
          LegalAction action;
          action.response_ = mask;
          legal_actions_.push_back(action);
          return;
        }
        for (int j = begin; j + remaining <= bits.size(); ++j)
          enumerate(j + 1, remaining - 1, mask | (uint32_t(1) << bits[j]));
      };
      enumerate(0, count, 0);
      to_play_ = player;
      callback_ = [this](int index) {
        YGO_SetResponsei(pduel_, legal_actions_[index].response_);
      };
''' + source[end:]
binding = env / "ygoenv/ygoenv/ygopro/ygopro.cpp"
text = subprocess.check_output(["git", "-C", str(env), "show", "HEAD:ygoenv/ygoenv/ygopro/ygopro.cpp"], text=True).replace('  m.def("init_module", &ygopro::init_module);', '''  m.def("init_module", &ygopro::init_module);
  m.def("integration_errors", []() { return ygopro::integration_errors; });
  m.def("prompt_counts", []() { return ygopro::integration_prompt_counts; });
  m.def("clear_prompt_counts", []() { ygopro::integration_prompt_counts.clear(); });
  m.attr("integration_revision") = 1;''')
from audit_patch import apply as apply_audit_patch
source, text = apply_audit_patch(source, text)
from resume_patch import apply as apply_resume_patch
source, text = apply_resume_patch(source, text)
if header.read_text() != source:
    header.write_text(source)
if binding.read_text() != text:
    binding.write_text(text)
interpreter = core / "interpreter.h"
text = subprocess.check_output(["git", "-C", str(core), "show", "HEAD:interpreter.h"], text=True).replace('#include <lua.h>', '#include <lua.hpp>')
if interpreter.read_text() != text:
    interpreter.write_text(text)
for filename in ("playerop.cpp", "processor.cpp", "operations.cpp"):
    path = core / filename
    original = subprocess.check_output(["git", "-C", str(core), "show", f"HEAD:{filename}"], text=True)
    text = '#include <iostream>\n' + original.replace('pduel->write_buffer8(MSG_RETRY);',
        'std::cerr << "CORE RESPONSE REJECTED " << __FILE__ << ":" << __LINE__ << std::endl; pduel->write_buffer8(MSG_RETRY);')
    if path.read_text() != text:
        path.write_text(text)
work = env / "scripts"
link = work / "script"
if not link.exists():
    link.symlink_to(scripts, target_is_directory=True)
dest = env / "assets/locale/en"
dest.mkdir(parents=True, exist_ok=True)
shutil.copy2(database, dest / "cards.cdb")
shutil.copy2(database.parent / "strings.conf", dest / "strings.conf")
with sqlite3.connect(database) as db:
    codes = [row[0] for row in db.execute("SELECT id FROM datas ORDER BY id")]
if len(codes) > 65534:
    raise RuntimeError("Card vocabulary exceeds native 16-bit representation")
(work / "code_list.txt").write_text("".join(f"{code} {int((scripts / f'c{code}.lua').exists())}\n" for code in codes))
manifest = json.loads((root / "engine/lock.json").read_text())
manifest.update(integration_revision=2, resume_rng_version=1, card_count=len(codes),
                database_sha256=hashlib.sha256(database.read_bytes()).hexdigest())
(env / "integration.json").write_text(json.dumps(manifest, indent=2))
print(f"Prepared source integration: {len(codes)} card definitions")
