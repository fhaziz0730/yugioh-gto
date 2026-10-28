"""Visibility fixes and an explicitly test-only deterministic board harness."""
def apply(source, binding):
    def replace(old, new):
        nonlocal source
        if old not in source:
            raise RuntimeError(f"Audit patch context changed: {old[:90]}")
        source = source.replace(old, new)

    replace('static std::string last_integration_response;', '''static std::string last_integration_response;
static bool audit_enabled = false;
static intptr_t audit_duel = 0;
static std::string audit_script;
''')
    replace('''    YGO_StartDuel(pduel, duel_options_);''', '''    if (audit_enabled) {
      cards_script_["./script/_audit.lua"] = {
        reinterpret_cast<byte*>(audit_script.data()), int(audit_script.size())};
      if (!preload_script(pduel, "./script/_audit.lua"))
        throw std::runtime_error("Audit scenario failed to load");
      std::vector<byte> discarded(0x100000);
      get_message(pduel, discarded.data());
      audit_duel = pduel;
    }
    YGO_StartDuel(pduel, duel_options_);''')
    # A reveal must not unlock unrelated private zones. Extra Decks must be
    # inspected card-by-card because face-up Pendulum monsters are public.
    replace('''        if (opponent && (revealed_.size() != 0)) {
          hidden_for_opponent = false;
        }''', '')
    replace('{LOCATION_EXTRA, true}', '{LOCATION_EXTRA, false}')
    # Revealed locations can move or be replaced. Do not use a stale location
    # to reveal the identity of a later facedown card.
    replace('''              if (revealed_.find(spec) != revealed_.end()) {
                hide = false;
              }''', '')
    replace('''      if (revealed_.size() != 0) {
        hidden_for_opponent = false;
      }''', '')
    replace('''      if (revealed_.find(spec) != revealed_.end()) {
        hide = false;
      }''', '')
    replace('''      if (!hide) {
        card_id = c_get_card_id(c.code_);
      }
    }
    return c_get_card_id(get_card_code(player, loc, seq));''', '''      if (!hide) {
        card_id = c_get_card_id(c.code_);
      }
      return card_id;
    }
    return c_get_card_id(get_card_code(player, loc, seq));''')
    # Persist legitimate reveal events as historical knowledge, without exposing
    # current hand slots, the rest of that hand, or any draw order.
    replace('''  std::unordered_set<std::string> revealed_;''', '''  std::unordered_set<std::string> revealed_;
  struct RevealEvent { CardId card; int turn; };
  std::array<std::vector<RevealEvent>, 2> reveal_events_;
  std::map<std::string, CardId> prompt_card_ids_;
''')
    replace('''    msg_ = int(data_[dp_++]);''', '''    prompt_card_ids_.clear();
    msg_ = int(data_[dp_++]);''')
    # Current cores send selection-local deck indices, not physical draw-order
    # indices. Preserve authorized IDs supplied with this selection prompt.
    start = source.index('    } else if (msg_ == MSG_SELECT_CARD) {')
    end = source.index('    } else if (msg_ == MSG_SELECT_TRIBUTE) {', start)
    segment = source[start:end]
    segment = segment.replace('''          specs.push_back(spec);''', '''          specs.push_back(spec);
          prompt_card_ids_[spec] = (card.controler_ == player ||
            (!(card.position_ & POS_FACEDOWN) && card.location_ != LOCATION_HAND && card.location_ != LOCATION_DECK && card.location_ != LOCATION_EXTRA))
            ? c_get_card_id(card.code_) : 0;''', 1)
    segment = segment.replace('''          dp_ += 4;
          auto controller = read_u8();''', '''          auto selection_code = read_u32();
          auto controller = read_u8();''')
    segment = segment.replace('''          auto spec = ls_to_spec(loc, seq, pos, controller != player);
          specs.push_back(spec);''', '''          auto spec = ls_to_spec(loc, seq, pos, controller != player);
          specs.push_back(spec);
          prompt_card_ids_[spec] = (controller == player ||
            (!(pos & POS_FACEDOWN) && loc != LOCATION_HAND && loc != LOCATION_DECK && loc != LOCATION_EXTRA))
            ? c_get_card_id(selection_code) : 0;''')
    source = source[:start] + segment + source[end:]
    replace('''        action.spec_index_ = spec_info.index;''', '''        action.spec_index_ = spec_info.index;
        auto selected = prompt_card_ids_.find(spec);
        if (selected != prompt_card_ids_.end()) action.cid_ = selected->second;''')
    replace('''  CardId spec_to_card_id(const std::string &spec, PlayerId player) {
    int offset = 0;''', '''  CardId spec_to_card_id(const std::string &spec, PlayerId player) {
    auto selected = prompt_card_ids_.find(spec);
    if (selected != prompt_card_ids_.end()) return selected->second;
    int offset = 0;''')
    replace('''    turn_count_ = 0;''', '''    turn_count_ = 0;
    revealed_.clear();
    reveal_events_[0].clear();
    reveal_events_[1].clear();''')
    replace('''"obs:global_"_.Bind(Spec<uint8_t>({23})),''', '''"obs:global_"_.Bind(Spec<uint8_t>({23})),
        "obs:reveals_"_.Bind(Spec<uint8_t>({32, 3})),''')
    replace('''    state["reward"_] = reward;''', '''    state["reward"_] = reward;
    auto& events = reveal_events_[to_play_];
    int event_count = std::min(int(events.size()), 32);
    for (int j = 0; j < event_count; ++j) {
      auto& event = events[events.size() - 1 - j];
      state["obs:reveals_"_](j, 0) = uint8_t(event.card >> 8);
      state["obs:reveals_"_](j, 1) = uint8_t(event.card);
      state["obs:reveals_"_](j, 2) = uint8_t(std::min(turn_count_ - event.turn + 1, 255));
    }''')
    replace('''    } else if (msg_ == MSG_CONFIRM_CARDS) {
      auto player = read_u8();
      read_u8(); // current core's skip-panel flag
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();''', '''    } else if (msg_ == MSG_CONFIRM_CARDS) {
      auto player = read_u8();
      read_u8(); // skip-panel flag
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        auto shown_code = read_u32();
        reveal_events_[player].push_back({c_get_card_id(shown_code), turn_count_});''')
    # Expose face-up Extra Deck positions; upstream encoded those as zero.
    replace('''        if (hide || (c.position_ & POS_FACEDOWN)) {
          f_cards(offset, 5) = position_to_id(POS_FACEDOWN);
        }''', '''        f_cards(offset, 5) = position_to_id(hide ? POS_FACEDOWN : c.position_);''')
    binding = '#include "ygopro-core/field.h"\n#include "ygopro-core/card.h"\n' + binding
    binding = binding.replace('  m.attr("integration_revision") = 1;', '''
  m.attr("integration_revision") = 2;
  m.def("set_audit_scenario", [](const std::string& script) {
    ygopro::audit_enabled = true;
    ygopro::audit_script = script;
  });
  m.def("audit_snapshot", []() {
    if (!ygopro::audit_enabled || !ygopro::audit_duel)
      throw std::runtime_error("Privileged snapshot is available only in audit scenarios");
    auto* pd = reinterpret_cast<duel*>(ygopro::audit_duel);
    pybind11::dict result;
    result["lp"] = std::vector<int>{pd->game_field->player[0].lp, pd->game_field->player[1].lp};
    std::vector<std::vector<uint32_t>> cards;
    for (auto* card : pd->cards) {
      if (!card->current.location) continue;
      cards.push_back({card->data.code, card->current.controler, card->current.location,
                       card->current.sequence, card->current.position});
    }
    result["cards"] = cards;
    result["chain_size"] = int(pd->game_field->core.current_chain.size());
    return result;
  });''')
    return source, binding
