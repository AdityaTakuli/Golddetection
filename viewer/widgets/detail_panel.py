import customtkinter
import threading
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from viewer.file_opener import open_file, reveal_in_folder


class DetailPanel(customtkinter.CTkFrame):
    """Right-side detail panel with dual images, event info, and retry button."""

    def __init__(self, parent, on_retry=None, **kwargs):
        width = kwargs.pop("width", 280)
        super().__init__(parent, width=width, corner_radius=0, **kwargs)
        self.grid_propagate(False)
        self._current_record = None
        self._image_refs = {}
        self._on_retry = on_retry

        # -- empty state --
        self.empty_label = customtkinter.CTkLabel(
            self, text="Select a detection\nto view details",
            font=customtkinter.CTkFont(size=12),
            text_color=("gray50", "gray60"),
        )
        self.empty_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=40)

        # -- content frame --
        self.content = customtkinter.CTkScrollableFrame(self, fg_color="transparent")

        # C270 Image
        customtkinter.CTkLabel(
            self.content, text="C270 Crop",
            font=customtkinter.CTkFont(size=10, weight="bold"),
            text_color=("gray40", "gray60"), anchor="w"
        ).pack(fill="x", padx=12, pady=(8, 2))

        self.c270_image_label = customtkinter.CTkLabel(
            self.content, text="pending...", height=120,
            fg_color=("gray92", "gray14"), corner_radius=8,
            cursor="hand2",
        )
        self.c270_image_label.pack(fill="x", padx=12, pady=(0, 2))
        self.c270_image_label.bind("<Button-1>", lambda e: self._open_c270_image())

        self.c270_path_label = customtkinter.CTkLabel(
            self.content, text="",
            font=customtkinter.CTkFont(size=9),
            text_color=("#1a6dba", "#7ec4ff"), anchor="w",
            cursor="hand2",
        )
        self.c270_path_label.pack(fill="x", padx=12, pady=(0, 4))
        self.c270_path_label.bind("<Button-1>", lambda e: self._open_c270_image())

        # Lenovo Image
        customtkinter.CTkLabel(
            self.content, text="Lenovo Frame",
            font=customtkinter.CTkFont(size=10, weight="bold"),
            text_color=("gray40", "gray60"), anchor="w"
        ).pack(fill="x", padx=12, pady=(4, 2))

        self.lenovo_image_label = customtkinter.CTkLabel(
            self.content, text="pending...", height=120,
            fg_color=("gray92", "gray14"), corner_radius=8,
            cursor="hand2",
        )
        self.lenovo_image_label.pack(fill="x", padx=12, pady=(0, 2))
        self.lenovo_image_label.bind("<Button-1>", lambda e: self._open_lenovo_image())

        self.lenovo_path_label = customtkinter.CTkLabel(
            self.content, text="",
            font=customtkinter.CTkFont(size=9),
            text_color=("#1a6dba", "#7ec4ff"), anchor="w",
            cursor="hand2",
        )
        self.lenovo_path_label.pack(fill="x", padx=12, pady=(0, 4))
        self.lenovo_path_label.bind("<Button-1>", lambda e: self._open_lenovo_image())

        # Separator
        customtkinter.CTkFrame(
            self.content, height=1, fg_color=("gray80", "gray30")
        ).pack(fill="x", padx=12, pady=8)

        # Detail fields
        self.details_frame = customtkinter.CTkFrame(self.content, fg_color="transparent")
        self.details_frame.pack(fill="x", padx=12, pady=(0, 8))

        # "Material" is the physics verdict; "Status" is only pipeline
        # completeness. A row can be fully processed and still not be gold.
        fields = ["Date", "Time", "Duration", "Weight", "Material",
                  "Confidence", "Sync", "Status", "Event ID"]
        self._detail_labels = {}
        for i, field in enumerate(fields):
            customtkinter.CTkLabel(
                self.details_frame, text=field,
                font=customtkinter.CTkFont(size=10),
                text_color=("gray50", "gray60"), anchor="w",
            ).grid(row=i, column=0, padx=(0, 8), pady=2, sticky="w")

            val = customtkinter.CTkLabel(
                self.details_frame, text="-",
                font=customtkinter.CTkFont(
                    size=12,
                    weight="bold" if field in ("Weight", "Status", "Material") else "normal",
                    family="Courier New" if field == "Event ID" else None,
                ),
                anchor="w",
            )
            val.grid(row=i, column=1, pady=2, sticky="w")
            self._detail_labels[field] = val

        self.details_frame.grid_columnconfigure(1, weight=1)

        # Error text (shown only for failed)
        self.error_label = customtkinter.CTkLabel(
            self.content, text="",
            font=customtkinter.CTkFont(size=10),
            text_color=("#B91C1C", "#FCA5A5"), anchor="w", wraplength=230,
        )
        self.error_label.pack(fill="x", padx=12, pady=(0, 4))

        # Separator
        customtkinter.CTkFrame(
            self.content, height=1, fg_color=("gray80", "gray30")
        ).pack(fill="x", padx=12, pady=8)

        # Action buttons
        self.btn_c270_video = customtkinter.CTkButton(
            self.content, text="Play C270 Video", height=34, corner_radius=8,
            command=self._play_c270,
        )
        self.btn_c270_video.pack(fill="x", padx=12, pady=(4, 3))

        self.btn_lenovo_video = customtkinter.CTkButton(
            self.content, text="Play Lenovo Video", height=34, corner_radius=8,
            fg_color=("gray70", "gray30"),
            command=self._play_lenovo,
        )
        self.btn_lenovo_video.pack(fill="x", padx=12, pady=3)

        self.btn_retry = customtkinter.CTkButton(
            self.content, text="Retry Processing", height=34, corner_radius=8,
            fg_color=("#854F0B", "#FAC775"),
            text_color=("white", "black"),
            command=self._retry,
        )
        self.btn_retry.pack(fill="x", padx=12, pady=3)

        self.btn_folder = customtkinter.CTkButton(
            self.content, text="Reveal in Folder", height=34, corner_radius=8,
            fg_color="transparent",
            border_width=1, border_color=("gray60", "gray40"),
            text_color=("gray20", "gray80"),
            hover_color=("gray85", "gray25"),
            command=self._reveal,
        )
        self.btn_folder.pack(fill="x", padx=12, pady=(3, 12))

    def load_record(self, record: dict):
        self._current_record = record
        self.empty_label.grid_forget()
        self.content.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)

        status = record.get("processing_status", "pending")

        # -- C270 Image --
        c270_img = record.get("image_c270")
        if c270_img and Path(c270_img).exists() and HAS_PIL:
            self._load_image_async(c270_img, self.c270_image_label, "c270")
            self.c270_path_label.configure(text=Path(c270_img).name)
        else:
            text = "pending..." if status in ("pending", "processing") else "No image"
            self.c270_image_label.configure(image=None, text=text)
            self.c270_path_label.configure(text=c270_img or "")
            self._image_refs.pop("c270", None)

        # -- Lenovo Image --
        len_img = record.get("image_lenovo")
        if len_img and Path(len_img).exists() and HAS_PIL:
            self._load_image_async(len_img, self.lenovo_image_label, "lenovo")
            self.lenovo_path_label.configure(text=Path(len_img).name)
        else:
            text = "pending..." if status in ("pending", "processing") else "No image"
            self.lenovo_image_label.configure(image=None, text=text)
            self.lenovo_path_label.configure(text=len_img or "")
            self._image_refs.pop("lenovo", None)

        # -- Details --
        captured = record.get("captured_at", "")
        try:
            dt = datetime.fromisoformat(captured)
            self._detail_labels["Date"].configure(text=dt.strftime("%d %b %Y"))
            self._detail_labels["Time"].configure(text=dt.strftime("%H:%M:%S"))
        except (ValueError, TypeError):
            self._detail_labels["Date"].configure(text="-")
            self._detail_labels["Time"].configure(text="-")

        dur = record.get("duration_sec")
        self._detail_labels["Duration"].configure(
            text=f"{dur:.1f}s" if dur else "-"
        )

        weight = record.get("weight")
        if weight and weight not in ("None", "unavailable", "", None):
            self._detail_labels["Weight"].configure(text=f"{weight} g",
                text_color=("#3B6D11", "#9FE1CB"))
        elif status == "done":
            self._detail_labels["Weight"].configure(text="none",
                text_color=("gray40", "gray60"))
        else:
            self._detail_labels["Weight"].configure(text="pending...",
                text_color=("#854F0B", "#FAC775"))

        # Confidence
        conf = record.get("detection_confidence")
        if conf is not None:
            self._detail_labels["Confidence"].configure(
                text=f"{conf:.2%}", text_color=("gray20", "gray80"))
        else:
            self._detail_labels["Confidence"].configure(
                text="pending..." if status in ("pending", "processing") else "-",
                text_color=("gray40", "gray60"))

        # Sync offset
        sync = record.get("sync_offset_ms")
        if sync is not None:
            self._detail_labels["Sync"].configure(
                text=f"{sync}ms", text_color=("gray20", "gray80"))
        else:
            self._detail_labels["Sync"].configure(
                text="-", text_color=("gray40", "gray60"))

        # Status badge color
        status_colors = {
            "pending":    ("#854F0B", "#FAC775"),
            "processing": ("#1a6dba", "#7ec4ff"),
            "done":       ("#3B6D11", "#9FE1CB"),
            "partial":    ("#7C5C00", "#FFD54F"),
            "failed":     ("#B91C1C", "#FCA5A5"),
        }
        sc = status_colors.get(status, ("gray40", "gray60"))
        self._detail_labels["Status"].configure(text=status, text_color=sc)

        # -- material verdict from the physics stage --
        verdict = record.get("material_verdict") or "-"
        vcolour = {
            "GOLD_LIKE":        ("#b8860b", "#ffd24a"),
            "NON_GOLD_METAL":   ("gray40", "gray70"),
            "DIELECTRIC":       ("#b02020", "#ff6b6b"),
            "UNCERTAIN":        ("#b06000", "#ffa94d"),
            "INVALID_CLIPPED":  ("#b06000", "#ffa94d"),
            "INVALID_NO_SWEEP": ("gray45", "gray65"),
        }.get(verdict, ("gray45", "gray65"))
        pretty = {
            "GOLD_LIKE":        "GOLD (optical)",
            "NON_GOLD_METAL":   "metal, not gold",
            "DIELECTRIC":       "NOT METAL",
            "UNCERTAIN":        "uncertain",
            "INVALID_CLIPPED":  "invalid - clipped",
            "INVALID_NO_SWEEP": "invalid - no sweep",
        }.get(verdict, verdict)
        self._detail_labels["Material"].configure(text=pretty, text_color=vcolour)

        eid = record.get("event_id", "-")
        self._detail_labels["Event ID"].configure(
            text=eid[:18] + ".." if len(eid) > 18 else eid
        )

        # Error / material notes. Both can be present, so neither is
        # allowed to hide the other.
        notes = []
        err = record.get("processing_error", "")
        if err:
            notes.append(f"Error: {err}")
        reason = record.get("material_reason") or ""
        if reason and verdict not in ("GOLD_LIKE", "-"):
            notes.append(f"Material: {reason}")
        self.error_label.configure(text="\n".join(notes))

        # Button states
        c270v = record.get("c270_video_path", "")
        lenv  = record.get("lenovo_video_path", "")
        self.btn_c270_video.configure(
            state="normal" if (c270v and Path(c270v).exists()) else "disabled"
        )
        self.btn_lenovo_video.configure(
            state="normal" if (lenv and Path(lenv).exists()) else "disabled"
        )
        # Retry only for pending/failed
        self.btn_retry.configure(
            state="normal" if status in ("pending", "failed") else "disabled"
        )

    def clear(self):
        self.content.grid_forget()
        self.empty_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=40)
        self._current_record = None

    def _play_c270(self):
        if self._current_record:
            open_file(self._current_record.get("c270_video_path", ""))

    def _play_lenovo(self):
        if self._current_record:
            open_file(self._current_record.get("lenovo_video_path", ""))

    def _open_c270_image(self):
        if self._current_record:
            p = self._current_record.get("image_c270", "")
            if p and Path(p).exists():
                open_file(p)

    def _open_lenovo_image(self):
        if self._current_record:
            p = self._current_record.get("image_lenovo", "")
            if p and Path(p).exists():
                open_file(p)

    def _retry(self):
        if self._current_record and self._on_retry:
            self._on_retry(self._current_record.get("event_id"))

    def _reveal(self):
        if self._current_record:
            path = (self._current_record.get("c270_video_path") or
                    self._current_record.get("image_c270") or "")
            if path:
                reveal_in_folder(path)

    def _load_image_async(self, image_path: str, label_widget, key: str):
        def _load():
            try:
                if not Path(image_path).exists():
                    return
                pil_img = Image.open(image_path)
                pil_img.thumbnail((240, 160), Image.LANCZOS)
                ctk_img = customtkinter.CTkImage(
                    light_image=pil_img, dark_image=pil_img,
                    size=(pil_img.width, pil_img.height),
                )
                self._image_refs[key] = ctk_img
                if label_widget.winfo_exists():
                    self.after(0, lambda: label_widget.configure(image=ctk_img, text=""))
            except Exception:
                pass
        threading.Thread(target=_load, daemon=True).start()
